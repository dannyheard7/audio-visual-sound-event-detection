import csv

def load_csv_as_list(filename):
        with open(filename, 'rt') as csvfile:
                data = csv.reader(csvfile)

                csv_list = [row for row in data]

        return csv_list