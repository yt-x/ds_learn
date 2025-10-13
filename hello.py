from pathlib import Path

p = Path("C:/project/data/report.txt")
new_p = p.with_suffix(".csv")

print(p)      # C:\project\data\report.txt
print(new_p)