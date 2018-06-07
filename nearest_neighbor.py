import numpy as np
import os
from scipy import misc

train_dir = "./dataset/dataset128/" #directory of training data for particular class
gen_dir = "./dataset/test_gen/"
num_samples = 30


def get_train_imgs():
	train_imgs = []
	for filename in os.listdir(train_dir):
	    if filename == ".DS_Store":
	        continue
	    img = misc.imread(train_dir + filename, mode='RGB')
	    flat = np.asarray(img).flatten()
	    train_imgs.append(flat)
	assert len(train_imgs) > 0
	train_imgs = np.vstack(train_imgs)
	#need to normalize??
	return train_imgs


def get_gen_imgs():
	num_imgs = 0.0
	generated_imgs = []
	#take first n images, or random sample with random.sample(the_list, 100)
	for filename in os.listdir(gen_dir): 
	    if num_imgs == num_samples: 
	    	break
	    if filename == ".DS_Store":
	        continue
	    img = misc.imread(gen_dir + filename, mode='RGB')
	    flat = np.asarray(img).flatten()
	    generated_imgs.append(flat)
	    num_imgs += 1

	assert len(generated_imgs) > 0
	generated_images = np.vstack(generated_imgs)
	#need to normalize??
	return num_imgs, generated_imgs



#select 100 random inages from the generated ones:


#generated_imgs = np.asarray(generated_imgs)
#generated_imgs = generated_imgs[:][:][:][3]
#generated_imgs = generated_imgs.T[:][:][:][:3].T #way to get around weird problem with array slicing


train_imgs = get_train_imgs()
num_gen, gen_imgs = get_gen_imgs()

#calculate nearest neighbor for each sample
total_dist = 0.0
for im in gen_imgs:
	nearest = None
	nearest_dist = float("inf")


	for i,t in enumerate(train_imgs):
		dist = np.sum(np.absolute(im - t))
		#print(str(i) + " " + str(dist))
		if dist < nearest_dist:
			nearest = i
			nearest_dist = dist
		
	print("nearest: " + str(nearest))
	total_dist += nearest_dist
	

mean_dist = total_dist/num_gen

print("Mean of nearest neighbors for " + str(num_gen) + " = " + str(mean_dist))

