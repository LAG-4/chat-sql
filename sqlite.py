import sqlite3

# Connect to sqlite
connection = sqlite3.connect("student.db")
cursor = connection.cursor()

# Create the table
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT (
    NAME    VARCHAR(25),
    CLASS   VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS   INT
);
"""
cursor.execute(table_info)

# Insert some records
records = [
    ("Aryan",   "AI/ML",        "A", 100),
    ("Saransh", "Data Science", "A",  90),
    ("Bhavya",  "French",       "B",  50),
    ("Aditya",  "AI/ML",        "A",  20),
    ("Ruchir",  "AI/ML",        "B", 100),
    ("Arihant", "French",       "B", 100),
]
cursor.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?)", records)

# Display all records
print("The inserted records are:")
for row in cursor.execute("SELECT * FROM STUDENT"):
    print(row)

# Commit and close
connection.commit()
connection.close()
