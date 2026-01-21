from pathlib import Path

path = Path(r"C:/Users/HOME/Desktop/Srbija1.0 aplikacija/Kalkulacije i kartice artikala -/kartice_zero_intervali_20260119191307.csv")
with path.open("r", encoding="utf-8", errors="ignore") as f:
    for i in range(5):
        line = f.readline()
        if not line:
            break
        print(line.strip())
