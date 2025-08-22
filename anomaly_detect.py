import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import joblib
import warnings
import json
from datetime import datetime
from collections import deque
warnings.filterwarnings('ignore')

# ======================
# 增强型周期计算函数
# ======================
def calculate_period(sampling_interval, physical_cycle):
    """根据物理周期自动计算STL参数"""
    # 确保最小数据量约束
    min_points = 2 * int(physical_cycle / sampling_interval)  
    
    # 周期点计算（保留整数）
    period_points = round(physical_cycle / sampling_interval)
    
    # 工业级校验
    if period_points < 4:
        raise ValueError(f"采样率不足！物理周期{physical_cycle}s需要"
                         f">={4*sampling_interval}Hz采样率")
    
    # 确保周期为奇数（STL内部要求）
    return int(period_points) if int(period_points) % 2 == 1 else int(period_points) + 1

# ======================
# 时序特征工程模块（工业增强版）
# ======================
class TimeSeriesFeatureEngineer:
    def __init__(self, physical_cycle=None, sampling_interval=None):
        """
        初始化特征工程参数
        物理周期（秒）和采样间隔（秒）共同确定周期参数
        """
        # 动态计算周期点
        if physical_cycle and sampling_interval:
            self.period = calculate_period(sampling_interval, physical_cycle)
            print(f"✅ 自动计算STL周期: {self.period}点 (物理周期={physical_cycle}s, 采样间隔={sampling_interval}s)")
        else:
            self.period = 1440  # 默认日周期（每分钟采样）
            print("⚠️ 使用默认周期1440点（日周期每分钟采样）")
        
        # 标准化工具
        self.scaler = StandardScaler()
        # 窗口大小历史记录（平滑用）
        self.window_history = deque(maxlen=5)
        
    def _stl_decomposition(self, values):
        """鲁棒STL时序分解获取残差"""
        try:
            # 检查数据长度是否足够
            if len(values) < 2 * self.period:
                raise ValueError(f"数据长度不足{2*self.period}点，无法进行STL分解")
                
            stl = STL(values, period=self.period, robust=True)
            res = stl.fit()
            return res.resid
        except Exception as e:
            print(f"⚠️ STL分解失败: {str(e)}，使用零残差替代")
            return np.zeros_like(values)
    
    def _safe_rolling(self, series, window, func):
        """带异常处理的滑动窗口计算"""
        try:
            # 窗口大小平滑
            self.window_history.append(window)
            smoothed_window = int(np.mean(self.window_history))
            
            # 检查窗口内NaN比例
            if series.isna().sum() > 0.3 * smoothed_window:
                return series.rolling(smoothed_window, min_periods=1).median()
            else:
                return func(series, smoothed_window)
        except Exception as e:
            print(f"⚠️ 滑动窗口计算失败: {str(e)}，使用中位数替代")
            return series.rolling(5, min_periods=1).median()

    def create_features(self, df):
        """
        输入: DataFrame包含timestamp和value列
        输出: 增强特征矩阵（已处理NaN）
        """
        # 克隆数据避免污染原始数据
        df = df.copy()
        
        # 基础特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # 滞后特征（工业优化：前向填充处理初始NaN）
        df['lag_1'] = df['value'].shift(1).fillna(method='bfill')
        df['lag_24'] = df['value'].shift(self.period).fillna(method='bfill')
        
        # 自适应窗口大小
        window_size = max(6, min(60, len(df)//10))  # 限制在6-60之间
        
        # 安全滑动窗口计算
        df['rolling_mean'] = self._safe_rolling(df['value'], window_size, 
                                               lambda s, w: s.rolling(w, min_periods=1).mean())
        df['rolling_std'] = self._safe_rolling(df['value'], window_size, 
                                              lambda s, w: s.rolling(w, min_periods=1).std())
        
        # STL残差特征
        df['residual'] = self._stl_decomposition(df['value'].values)
        
        # 周期编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
        df['week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
        
        # 特征选择
        features = df[['value', 'lag_1', 'lag_24', 'rolling_mean', 
                      'rolling_std', 'residual', 'hour_sin', 'hour_cos',
                      'week_sin', 'week_cos']]
        
        # ==== 工业级NaN处理 ====
        # 1. 替换无穷值
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # 2. 记录初始NaN分布
        initial_nan = features.isna().sum().sum()
        
        # 3. 分类型填充策略
        for col in features.columns:
            col_data = features[col]
            nan_count = col_data.isna().sum()
            
            if nan_count == 0:
                continue
                
            # 根据特征类型选择填充策略
            if 'lag' in col:
                features[col] = col_data.fillna(method='bfill')
            elif 'rolling' in col:
                features[col] = col_data.interpolate(method='linear')
            elif 'residual' in col:
                features[col] = col_data.fillna(0)  # 残差缺失视为正常
            else:
                features[col] = col_data.fillna(col_data.median())
        
        # 4. 最终检查
        final_nan = features.isna().sum().sum()
        if final_nan > 0:
            print(f"⚠️ 警告：残留 {final_nan} 个NaN，使用全局中位数填充")
            features = features.fillna(features.median())
        
        # 5. 报告数据质量
        if initial_nan > 0:
            print(f"🛠 处理NaN：初始{initial_nan} → 处理后{final_nan}")
        
        # 标准化（仅用有效数据训练）
        if not hasattr(self, 'fitted_scaler'):
            valid_features = features.dropna()
            if len(valid_features) < 100:
                print("❌ 有效数据不足100点，无法训练标准化器")
                return None
            self.fitted_scaler = self.scaler.fit(valid_features)
        
        return self.fitted_scaler.transform(features)

# ======================
# 动态OCSVM模型（工业增强版）
# ======================
class DynamicOCSVM:
    def __init__(self, nu=0.03, gamma='scale', update_interval=24):
        """
        nu: 预期异常比例
        gamma: RBF核参数
        update_interval: 模型更新间隔(小时)
        """
        self.nu = nu
        self.gamma = gamma
        self.update_interval = update_interval
        self.last_update_time = None
        self.model = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        # 支持向量缓存
        self.support_vectors = None
        
    def needs_update(self, current_time):
        """检查是否需要更新模型"""
        if self.last_update_time is None:
            return True
        hours_passed = (current_time - self.last_update_time).total_seconds()/3600
        return hours_passed >= self.update_interval
    
    def _nan_safety_check(self, X):
        """工业级NaN检查与处理"""
        if X is None:
            raise ValueError("输入数据为空")
            
        nan_mask = np.isnan(X)
        nan_count = np.sum(nan_mask)
        
        if nan_count > 0:
            nan_percentage = 100 * nan_count / X.size
            print(f"⚠️ 模型输入包含 {nan_count} 个NaN ({nan_percentage:.2f}%)")
            
            # 简单填充策略
            col_medians = np.nanmedian(X, axis=0)
            for i in range(X.shape[1]):
                X[np.isnan(X[:, i]), i] = col_medians[i]
                
        return X
    
    def train(self, X):
        """鲁棒训练方法"""
        # 数据质量检查
        X = self._nan_safety_check(X)
        
        print(f"⏳ 初始模型训练中，样本数: {len(X)}")
        try:
            self.model.fit(X)
            self.last_update_time = pd.Timestamp.now()
            self.support_vectors = self.model.support_vectors_.copy()
            print(f"✅ 训练成功，支持向量数: {len(self.support_vectors)}")
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            # 应急方案：创建虚拟模型
            self.model = OneClassSVM(nu=self.nu, kernel='rbf', gamma=self.gamma)
            dummy_data = np.random.randn(100, X.shape[1])
            self.model.fit(dummy_data)
            self.support_vectors = dummy_data
            print("⚠️ 已加载应急模型")
        
        return self
    
    def partial_update(self, X_new):
        """增量更新模型"""
        if len(X_new) < 50:  # 降低样本要求
            print(f"⚠️ 新增样本不足({len(X_new)}<50)，跳过更新")
            return
            
        print(f"🔄 模型增量更新中，新增样本: {len(X_new)}")
        
        try:
            # 合并新旧数据（保留历史支持向量）
            X_combined = np.vstack([self.support_vectors, X_new])
            
            # 创建新模型
            new_model = OneClassSVM(nu=self.nu, kernel='rbf', gamma=self.gamma)
            new_model.fit(X_combined)
            
            # 更新状态
            self.model = new_model
            self.support_vectors = new_model.support_vectors_.copy()
            self.last_update_time = pd.Timestamp.now()
            print(f"✅ 更新完成，支持向量数: {len(self.support_vectors)}")
        except Exception as e:
            print(f"❌ 增量更新失败: {str(e)}，保留旧模型")

    def predict(self, X, threshold=0.0):
        """预测异常分数和标签"""
        # 数据质量检查
        X = self._nan_safety_check(X)
        
        try:
            scores = -self.model.decision_function(X)
            return scores, scores > threshold
        except Exception as e:
            print(f"❌ 预测失败: {str(e)}，返回零分数")
            return np.zeros(len(X)), np.zeros(len(X), dtype=bool)

# ======================
# 异常检测系统主流程（工业级）
# ======================
class IndustrialAnomalyDetector:
    def __init__(self, physical_cycle=86400, sampling_interval=300):
        """
        physical_cycle: 物理周期（秒），如86400=24小时
        sampling_interval: 采样间隔（秒），如300=5分钟
        """
        self.feature_engineer = TimeSeriesFeatureEngineer(
            physical_cycle=physical_cycle,
            sampling_interval=sampling_interval
        )
        self.detector = DynamicOCSVM()
        self.data_buffer = pd.DataFrame(columns=['timestamp', 'value'])
        self.threshold_history = []
        # 数据质量监控
        self.data_quality_log = []
        
    def _log_data_quality(self, nan_count, total_points):
        """记录数据质量问题"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "nan_count": nan_count,
            "total_points": total_points,
            "nan_percentage": 100 * nan_count / total_points
        }
        self.data_quality_log.append(entry)
        
        # 严重问题告警
        if entry["nan_percentage"] > 20:
            print(f"🚨 严重数据质量问题！NaN比例: {entry['nan_percentage']:.1f}%")
    
    def process_stream(self, new_data):
        """
        处理实时数据流
        输入: DataFrame 包含 ['timestamp', 'value']
        """
        # 0. 输入验证
        if new_data.isnull().any().any():
            nan_count = new_data.isnull().sum().sum()
            self._log_data_quality(nan_count, new_data.size)
        
        # 1. 数据缓冲（带时间排序）
        self.data_buffer = pd.concat([self.data_buffer, new_data])
        self.data_buffer.sort_values('timestamp', inplace=True)
        self.data_buffer.drop_duplicates('timestamp', inplace=True)
        
        # 2. 特征工程（已内置NaN处理）
        current_time = new_data['timestamp'].max()
        features = self.feature_engineer.create_features(self.data_buffer.copy())
        
        # 特征工程失败处理
        if features is None:
            print("❌ 特征工程失败，返回默认结果")
            return np.zeros(len(new_data)), np.zeros(len(new_data), dtype=bool)
        
        # 3. 模型初始训练（首次运行）
        if not hasattr(self.detector.model, 'fit_status_'):
            min_data_size = max(100, 2 * self.feature_engineer.period)
            if len(self.data_buffer) < min_data_size:
                print(f"⌛ 等待数据累积 ({len(self.data_buffer)}/{min_data_size})")
                return np.zeros(len(new_data)), np.zeros(len(new_data), dtype=bool)
            
            print(f"🚀 开始初始训练，数据量: {len(features)}")
            self.detector.train(features)
        
        # 4. 动态模型更新
        if self.detector.needs_update(current_time):
            recent_data = self.data_buffer[self.data_buffer['timestamp'] >= self.detector.last_update_time]
            if len(recent_data) > 0:
                print(f"⏱ 触发模型更新，新数据量: {len(recent_data)}")
                recent_features = self.feature_engineer.create_features(recent_data)
                if recent_features is not None:
                    self.detector.partial_update(recent_features)
        
        # 5. 实时预测（仅对新数据）
        new_indices = self.data_buffer.index[-len(new_data):]
        new_features = features[new_indices]
        scores, labels = self.detector.predict(new_features)
        
        # 6. 动态阈值调整
        try:
            current_threshold = self._calculate_dynamic_threshold(scores)
            refined_labels = scores > current_threshold
        except Exception as e:
            print(f"❌ 阈值计算失败: {str(e)}，使用固定阈值")
            refined_labels = scores > np.percentile(scores, 95)
        
        return scores, refined_labels
    
    def _calculate_dynamic_threshold(self, scores):
        """基于历史数据的动态阈值计算"""
        # 获取当前小时
        current_hour = pd.Timestamp.now().hour
        
        # 更新阈值历史记录
        hour_scores = {'hour': current_hour, 'scores': scores}
        self.threshold_history.append(hour_scores)
        
        # 保留最近7天数据
        if len(self.threshold_history) > 168:  # 24*7
            self.threshold_history.pop(0)
        
        # 计算当前时段的统计量
        hour_data = [s for h in self.threshold_history 
                    if h['hour'] == current_hour for s in h['scores']]
        
        if len(hour_data) > 20:  # 提高最小样本要求
            q75 = np.percentile(hour_data, 75)
            q25 = np.percentile(hour_data, 25)
            iqr = q75 - q25
            
            # IQR保护机制
            iqr = max(iqr, 0.01 * (np.max(hour_data) - np.min(hour_data)))
            
            threshold = q75 + 1.5 * iqr
            print(f"📊 动态阈值: {threshold:.4f} (Q75={q75:.4f}, IQR={iqr:.4f})")
        else:
            threshold = np.percentile(scores, 95)
            print(f"📊 初始阈值: {threshold:.4f} (P95)")
        
        return threshold

    def save_state(self, file_path):
        """保存系统状态"""
        state = {
            'data_buffer': self.data_buffer,
            'threshold_history': self.threshold_history,
            'data_quality_log': self.data_quality_log,
            'feature_engineer': self.feature_engineer,
            'detector': self.detector
        }
        joblib.dump(state, file_path)
        print(f"💾 系统状态已保存至 {file_path}")
    
    def load_state(self, file_path):
        """加载系统状态"""
        state = joblib.load(file_path)
        self.data_buffer = state['data_buffer']
        self.threshold_history = state['threshold_history']
        self.data_quality_log = state['data_quality_log']
        self.feature_engineer = state['feature_engineer']
        self.detector = state['detector']
        print(f"🔍 系统状态已从 {file_path} 加载")

# ======================
# 使用示例（工业级测试）
# ======================
if __name__ == "__main__":
    # 1. 模拟时序数据生成（带NaN模拟）
    def generate_industrial_data(num_points=5000, anomaly_rate=0.01, nan_rate=0.03):
        base_time = pd.Timestamp.now() - pd.Timedelta(days=7)
        timestamps = pd.date_range(start=base_time, periods=num_points, freq='5min')
        
        # 基础信号：趋势+周期
        trend = np.linspace(0, 5, num_points)
        seasonal = 3 * np.sin(2 * np.pi * np.arange(num_points)/288)  # 日周期(24h/5min=288)
        
        # 添加噪声
        values = trend + seasonal + np.random.normal(0, 0.8, num_points)
        
        # 插入人工异常
        anomaly_count = int(num_points * anomaly_rate)
        anomaly_points = np.random.choice(num_points, size=anomaly_count, replace=False)
        values[anomaly_points] += np.random.uniform(4, 10, anomaly_count)
        
        # 插入NaN值
        nan_points = np.random.choice(num_points, size=int(num_points * nan_rate), replace=False)
        values[nan_points] = np.nan
        
        df = pd.DataFrame({'timestamp': timestamps, 'value': values})
        
        # 前向填充NaN
        df['value'] = df['value'].fillna(method='ffill')
        return df
    
    # 2. 初始化检测系统（物理周期=24小时，采样间隔=5分钟）
    detector = IndustrialAnomalyDetector(
        physical_cycle=86400,  # 24小时物理周期（秒）
        sampling_interval=300   # 5分钟采样间隔（秒）
    )
    
    # 3. 模拟数据流处理
    all_data = generate_industrial_data()
    results = []
    
    # 分批处理模拟实时流
    batch_size = 12  # 每小时处理一次（12个5分钟点）
    for i in range(0, len(all_data), batch_size):
        batch = all_data.iloc[i:i+batch_size]
        
        print(f"\n=== 处理批次 {i//batch_size+1}/{(len(all_data)//batch_size)} ===")
        print(f"时间范围: {batch['timestamp'].min()} 至 {batch['timestamp'].max()}")
        
        # 核心检测调用
        scores, labels = detector.process_stream(batch)
        
        # 存储结果
        batch_results = batch.copy()
        batch_results['anomaly_score'] = scores
        batch_results['is_anomaly'] = labels
        results.append(batch_results)
        
        # 打印异常检测结果
        if np.any(labels):
            print(f"🚩 检测到异常: {np.sum(labels)}个")
            anomalies = batch_results[labels]
            print(anomalies[['timestamp', 'value', 'anomaly_score']])
    
    # 4. 结果分析
    final_results = pd.concat(results)
    anomalies = final_results[final_results['is_anomaly']]
    
    print(f"\n🔍 检测完成! 共发现 {len(anomalies)} 个异常点")
    print("最近5个异常点:")
    print(anomalies[['timestamp', 'value', 'anomaly_score']].tail(5))
    
    # 5. 系统状态保存
    detector.save_state('industrial_detector_state.pkl')