import numpy as np
import os, sys
import torch
from scipy import misc
from datamanager import GoogleLandmark
from torch.utils.data import DataLoader
from torchvision import transforms

# python nearest_neighbor.py [train_dir] wgang/hyperparam-... 5554

#train_dir = "./dataset/dataset128/" #directory of training data for particular class
#gen_dir = "./dataset/test_gen/"
num_samples = 30


def get_train_imgs(train_dir, image_dim, class_id):
	device = torch.device('cuda')
	dataset = GoogleLandmark(train_dir, [class_id],
                   transform=lambda c: transforms.Compose([
			   transforms.CenterCrop(c),
			   transforms.Resize(image_dim),
			   transforms.ToTensor(),
                  ]))
	data_loader = DataLoader(dataset, #batch_size=args['batch_size'],
                shuffle=True, drop_last=True, pin_memory=True)
	"""
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
	#need to normalize??
	return data_loader


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
	return generated_imgs



#select 100 random inages from the generated ones:


#generated_imgs = np.asarray(generated_imgs)
#generated_imgs = generated_imgs[:][:][:][3]
#generated_imgs = generated_imgs.T[:][:][:][:3].T #way to get around weird problem with array slicing

if __name__ == '__main__':
	train_dir = sys.argv[1]
	gen_dir = sys.argv[2]
	image_dim = int(sys.argv[3])
	class_id = int(sys.argv[4])
	train_loader = get_train_imgs(train_dir, image_dim, class_id)
	gen_imgs = get_gen_imgs(gen_dir)
	print("calculating nearest neighbors...")

	#calculate nearest neighbor for each sample
	total_dist = 0.0
	for im in gen_imgs:
		im = torch.tensor(im).cuda()
		#print("img: ", im.shape)
		nearest = None
		nearest_dist = float("inf")


		for size, t in enumerate(train_loader):
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

