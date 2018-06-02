# !/usr/bin/python
# Usage: python hydrate-data.py data_file output_dir [--num_cpus N] [--noheader]

import argparse, csv, os, platform, subprocess, sys
import multiprocessing as mp
from functools import partial
from io import BytesIO
from threading import Lock
from urllib import request, error
from PIL import Image
import tqdm

DEFAULT_NUM_CPUS = 4
MIN_GIGS_REQUIRED = 75
CHUNK_SIZE = 7

def parse_data(data_file, has_header):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  data = [line for line in csvreader]
  return data[1:] if has_header else data  # Chop off header if needed

def download_image(out_dir, datum):
  (key, url, landmark) = datum
  filename = os.path.join(out_dir, '{}-{}.jpg'.format(key, landmark))

  if os.path.exists(filename):
    errors.append('Image {} already exists. Skipping download.'.format(filename))
    return

  try:
    response = request.urlopen(url)
    image_data = response.read()
  except:
    errors.append('Warning: Could not download image {} from {}'.format(key, url))
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    errors.append('Warning: Failed to parse image {}'.format(key))
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    errors.append('Warning: Failed to convert image {} to RGB'.format(key))
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90) # 90% compression quality
  except:
    errors.append('Warning: Failed to save image {}'.format(filename))
    return

def report_stats(filename, num_imgs, errors):
  num_bad = len(errors)
  num_good = num_imgs - num_bad
  print("Successfully downloaded {}/{} images.".format(num_good, num_imgs))
  print("Had errors downloading {} images.".format(num_bad))
  basename = os.path.splitext(os.path.basename(filename))[0]
  with open('errors-{}.txt'.format(basename), 'w') as f:
    for error in errors:
      f.write(error + '\n')
  print("More details on errors can be found in errors-{}.txt".format(basename))

def init(error_list):
  global errors
  errors = error_list

def loader(data_file, out_dir, num_cpus, has_header):
  if not os.path.exists(out_dir):
    os.mkdir(out_dir)
  data = parse_data(data_file, has_header)

  with mp.Manager() as manager:
    errors = manager.list()
    pool = mp.Pool(processes=num_cpus, initializer=init, initargs=(errors, ))
    func = partial(download_image, out_dir)
    for _ in tqdm.tqdm(pool.imap(func, data, CHUNK_SIZE), total=len(data)):
      pass
    pool.close()
    pool.terminate()

    report_stats(data_file, len(data), errors)

if __name__ == '__main__':
  if os.name == 'posix' and platform.system() == 'Linux':
    stats = subprocess.check_output(['df', "-h", '/dev/sda1']).decode('utf-8')
    stats = [row.split() for row in stats.split('\n')]
    gigs_left = int(stats[1][stats[0].index("Avail")][:-1])
    if gigs_left < MIN_GIGS_REQUIRED:
      print("You may not have enough disk space to hold all the images. Please increase your disk size. Aborting.")
      exit()
  parser = argparse.ArgumentParser(description='Download images for the Google Landmarks Dataset.')
  parser.add_argument('data_file', type=str, help='A .csv file containing image id, image url, and landmark id.')
  parser.add_argument('output_dir', type=str, help='Directory to place retrieved images.')
  parser.add_argument('--num_cpus', type=int, dest='num_cpus', default=DEFAULT_NUM_CPUS, help='Number of available CPUs.')
  parser.add_argument('--noheader', action='store_false', dest='has_header', help='Flag indicating the .csv file has no header.')
  args = parser.parse_args()
  loader(args.data_file, args.output_dir, args.num_cpus, args.has_header)
