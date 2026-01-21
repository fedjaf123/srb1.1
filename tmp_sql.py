import sqlite3
from pathlib import Path

path = Path(r"C:/Users/HOME/Desktop/Srbija1.0 aplikacija/SRB1.0 - Copy.db")
conn = sqlite3.connect(path)
cur = conn.cursor()

cur.execute("SELECT sku, artikal, zero_from, zero_to, days_without FROM cartice_zero_intervals LIMIT 5")
print("zero intervals sample:")
for row in cur.fetchall():
    print(row)

cur.execute(
    "SELECT sku, artikal, total_sold, avg_daily, lost_qty "
    "FROM prodaja_stats ORDER BY lost_net DESC LIMIT 5"
)
print("\nprodaja stats sample:")
for row in cur.fetchall():
    print(row)

conn.close()
