import sys, os
from PIL import Image


if __name__ == '__main__':
  in_dir = sys.argv[1]
  out_dir = sys.argv[2]
  for filename in os.listdir(in_dir):
    if filename.endswith(".jpg"):
      outfile = out_dir + filename
      img = Image.open(in_dir + '/' + filename)
      img = img.resize((128,128), Image.ANTIALIAS)
      img.save(outfile)


