# !/usr/bin/python
# Usage: python split-csv.py csvfile num_lines [--noheader]

import argparse, os

def splitter(filename, lines_per_file, has_header):
  with open(filename, 'r') as fr:
    stripped_name = os.path.splitext(filename)[0]
    if has_header:
      header_line = fr.readline()
    file_num = 0
    while True:
      with open("{}-{}.csv".format(stripped_name, file_num), 'w') as fw:
        if has_header:
          fw.write(header_line)
        for i in range(lines_per_file):
          line = fr.readline()
          if line == "": return
          fw.write(line)
      file_num += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Split a large CSV file.')
  parser.add_argument('csvfile', type=str, help='A .csv file.')
  parser.add_argument('num_lines', type=int, help='Number of non-header lines to put in each file.')
  parser.add_argument('--noheader', action='store_false', dest='has_header', help='Flag indicating the .csv file has no header.')
  args = parser.parse_args()
  splitter(args.csvfile, args.num_lines, args.has_header)
