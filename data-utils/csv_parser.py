### USAGE: python download_data.py <data_file.csv> <output_file> <landmark_id> ###


import sys, os, csv


if __name__ == '__main__':
  if len(sys.argv) != 4:
    print('Syntax: %s <data_file.csv> <output_file> <landmark_id>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_file, landmark_id) = sys.argv[1:]

  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  first_line = True
  with open(out_file, 'w+') as f:
    for line in csvfile:
      if first_line:
        first_line = False
        continue
      _, _, lid = line.split(',')
      if int(lid) == int(landmark_id):
        f.write(line)
