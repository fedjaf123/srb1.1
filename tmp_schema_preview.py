from pathlib import Path

lines = Path("SRB1.1.py").read_text(encoding="utf-8").splitlines()
for i in range(640, 710):
    print(f"{i+1}: {lines[i]}")
