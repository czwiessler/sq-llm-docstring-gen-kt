import csv
import sys


def read_data(filename):
    """Read in the n top performing results from a given file"""
    results = []
    with open(filename) as data:
        reader = csv.reader(data)
        for line in reader:
            results.append((float(line[-2]), line[-1]))
    return results


if __name__ == "__main__":
    data_filename = sys.argv[1]
    results = read_data(data_filename)
    results.sort()
    for result in results[-10:]:
        print(result[0], result[1])
