import csv

with open('iris.csv') as inp, open('iris_id.csv', 'w') as out:
    reader = csv.reader(inp)
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['Id'] + next(reader))
    writer.writerows([i] + row for i, row in enumerate(reader, 1))
