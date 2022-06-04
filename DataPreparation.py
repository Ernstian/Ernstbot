import sqlite3


Brett = "b"
MinLength = 100




Database = "Pfosten.db"
connection = sqlite3.connect(Database)
cursor = connection.cursor()
executestring = "SELECT Text FROM Pfosten WHERE Brett = '" + str(Brett) + "'"
cursor.execute(executestring)
RawResults = cursor.fetchall()

import re
Dataset = []

for Result in RawResults:
    WorkString = Result[0]
    WorkString = re.sub(r'\\n', '', WorkString)
    WorkString = re.sub(r'\\t', '', WorkString)
    WorkString = re.sub(r'(>>\d*)', '', WorkString)
    FinishedString = WorkString.strip()
    if len(FinishedString)>MinLength:
        Dataset.append(FinishedString[1:-1])


with open("Dataset.txt", "w", encoding = "utf-8") as file:
    for Pfosten in Dataset:
        file.write(Pfosten)
        file.write("\n")

