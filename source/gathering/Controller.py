import psycopg2

conn = psycopg2.connect("dbname=postgres user=postgres password=280505")

cursor = conn.cursor()

cursor.execute("SELECT * FROM")