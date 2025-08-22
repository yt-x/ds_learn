# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:17:33 2025

@author: lch99
"""

def efficiency(ps, acs, ws):
    ef = acs / ps*100
    uph = acs / ws
    return round(ef, 2), round(uph, 2)

efficiency, rate = efficiency (ps=500, acs=450, ws=8)   
print(f"生产效率: {efficiency}%，每小时产量: {rate}辆")
