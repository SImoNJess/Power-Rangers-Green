import sqlite3
import os

# Path to your SQLite database
db_path = os.path.join(os.path.dirname(__file__), 'smartgrid.db')

# Output file
output_file = os.path.join(os.path.dirname(__file__), 'history_full.txt')

# Connect to the database
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Query: include day, tick, sun, demand
cur.execute(
    """
    SELECT day, tick, sun, demand
      FROM smart_grid_data
     ORDER BY day, tick
    """
)
rows = cur.fetchall()
conn.close()

# Write to text file
with open(output_file, 'w') as f:
    # Header
    f.write('day tick sun demand\n')
    # Each row
    for day, tick, sun, demand in rows:
        f.write(f"{day} {tick} {sun} {demand}\n")

print(f"Exported {len(rows)} rows to {output_file}")
