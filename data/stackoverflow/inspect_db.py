# inspect_db.py
import os, sqlite3, sys

db = sys.argv[1] if len(sys.argv) > 1 else "/path/to/stackoverflow.sqlite"
uri = f"file:{os.path.abspath(db)}?mode=ro"
conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
cur = conn.cursor()

print("== tables ==")
for (t,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"):
    print(" ", t)
print()

# 找潜在包含 client_id 的表
cands = []
for (t,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table';"):
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info('{t}')").fetchall()]
    if any(c.lower() in ("client_id","user_id") for c in cols):
        cands.append((t, cols))
        print(f"table: {t}  cols: {cols}")

print("\n== sample_embedded_data rows (first 5) from candidates ==")
for t, cols in cands:
    try:
        q = f"SELECT * FROM {t} LIMIT 5"
        rows = cur.execute(q).fetchall()
        print(f"\n-- {t} --")
        print(cols)
        for r in rows:
            print(r)
    except Exception as e:
        print(f"[skip {t}] {e}")

conn.close()
