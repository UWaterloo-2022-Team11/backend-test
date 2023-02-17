import json
import psycopg2
import pickle
import numpy as np

db_con = {}
with open('db_con.json', 'r') as f:
    db_con = json.load(f)

def decode_hex(text):
    return bytes.fromhex(text[2:]).decode('ascii')[1:-1]

conn = psycopg2.connect(database="postgres",
                        host=db_con['host'],
                        user=db_con['user'],
                        password=db_con['password'],
                        port=db_con['port'])

cur = conn.cursor()

cur.execute("select * from public.\"PinData\" where \"vector\" is not null;")

# 924082417262995924, 'img', 'link', 'uname', <memory at 0x000002175D47BDC0>, True
rows = cur.fetchall()
data = []
for row in rows:
    data.append([row[0], decode_hex(row[1]), decode_hex(row[2]), row[3], np.frombuffer(row[4], dtype=np.float32)])

with open('new_data.pkl', 'wb') as f:
    pickle.dump(data, f)