import numpy as np
import random
import glob
import os, sys
import torch
from scipy import misc
from datamanager import GoogleLandmark
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# python nearest_neighbor.py [train_dir] wgang/hyperparam-... 5554

#train_dir = "./dataset/dataset128/" #directory of training data for particular class
#gen_dir = "./dataset/test_gen/"
num_samples = 30


def get_train_imgs(train_dir, image_dim, class_id):
  device = torch.device('cuda')
  #total_size = len(os.listdir(train_dir))
  """
  dataset = GoogleLandmark(train_dir, [class_id],
	     transform=lambda c: transforms.Compose([
		     transforms.CenterCrop(c),
		     transforms.Resize(image_dim),
		     transforms.ToTensor(),
	    ]))
  data_loader = DataLoader(dataset,# batch_size=total_size,
	  shuffle=False, pin_memory=True)
  train_imgs = []
  for filename in os.listdir(train_dir):
      if filename == ".DS_Store":
	  continue
      img = misc.imread(train_dir + '/' + filename, mode='RGB')
      flat = np.asarray(img).flatten()
      train_imgs.append(flat)
  assert len(train_imgs) > 0
  train_imgs = np.vstack(train_imgs)
  """
  filenames = glob.glob(os.path.join(train_dir, '*-{}.jpg'.format(class_id)))
  images = []
  # If on-demand data loading
  
  it = 0
  for fname in filenames:
    if random.randint(0, 50) != 0:
      continue
    if it % 1000 == 0:
      print("openin", fname)
    it += 1
    image = Image.open(fname)
      
    # May use transform function to transform samples
    # e.g., random crop, whitening
    transform=lambda c: transforms.Compose([
	     transforms.CenterCrop(c),
	     transforms.Resize(image_dim),
	     transforms.ToTensor(),
    ])
    image = transform(min(image.size))(image)
    # return image and label
    images.append(image)
  #need to normalize??
  return torch.stack(images)


def get_gen_imgs(gen_dir): # TODO
  num_imgs = 0.0
  generated_imgs = []
  #take first n images, or random sample with random.sample(the_list, 100)
  for filename in os.listdir(gen_dir): 
      if num_imgs == num_samples: 
        break
      if filename == ".DS_Store":
          continue
      #img = misc.imread(gen_dir + '/' +  + filename, mode='RGB')
      generated_imgs.append(torch.load(gen_dir + '/' + filename))
      num_imgs += 1

  assert len(generated_imgs) > 0
  #need to normalize??
  return torch.stack(generated_imgs)

#select 100 random inages from the generated ones:
#generated_imgs = np.asarray(generated_imgs)
#generated_imgs = generated_imgs[:][:][:][3]
#generated_imgs = generated_imgs.T[:][:][:][:3].T #way to get around weird problem with array slicing

if __name__ == '__main__':
  train_dir = sys.argv[1]
  gen_dir = sys.argv[2]
  image_dim = int(sys.argv[3])
  class_id = int(sys.argv[4])
  train_set = get_train_imgs(train_dir, image_dim, class_id)
  gen_imgs = get_gen_imgs(gen_dir)#[0]
  print("calculating nearest neighbors...")

  #calculate nearest neighbor for each sample
  train_set = train_set.cuda()
  g = gen_imgs.shape[0]
  gen_imgs = gen_imgs.reshape(g,3*64*64)
  x = train_set.shape[0]
  train_set = train_set.reshape(x,3*64*64)
  print("gen: ", gen_imgs.shape)
  print("train: ", train_set.shape)
  total_dists = 0.0
  for i in range(g):  # for each generated image, find its closest neighbor in train set
    diffs = (gen_imgs[i] - train_set).abs_()
    #print("diff shape: ", diffs.shape)
    sums = diffs.sum(1)
    #print("sum shape: ", sums.shape)
    dist = torch.min(sums).data.cpu().numpy()
    total_dists += dist
  mean_dist = total_dists / g

  print("Mean of nearest neighbors for " + str(g) + " = " + str(mean_dist))
  """
  for im in gen_imgs:
    im = torch.tensor(im).cuda()
    #print("img: ", im.shape)
    nearest = None
    nearest_dist = float("inf")


    for size, t in enumerate(train_set):
      t = t[0][0].cuda()
      #print("imt: ", t.shape)
      dist = (im - t).abs_().sum()
      #print(str(i) + " " + str(dist))
      if dist < nearest_dist:
        #nearest = t
        nearest_dist = dist
      
    #print("nearest: " + str(nearest))
    total_dist += nearest_dist
    
  num_gen = len(gen_ims)
  mean_dist = total_dist/num_gen

  print("Mean of nearest neighbors for " + str(num_gen) + " = " + str(mean_dist))
  """

