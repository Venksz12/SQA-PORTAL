import sqlite3

conn = sqlite3.connect("sqa_dual_portal.db")

conn.execute("""
DELETE FROM users
WHERE username IN
(
'tester',
'dealer',
'company',
'auditor'
)
""")

conn.commit()
conn.close()

print("Demo users removed")