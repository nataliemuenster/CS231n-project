#Refined from original paper: https://arxiv.org/pdf/1801.01973.pdf
#https://github.com/sbarratt/inception-score-pytorch

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import os
from scipy import misc

import torchvision.models

import numpy as np
from scipy.stats import entropy

img_dir = "../dataset/good-pngs/"#out128-noise_dim32/"

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: if you want to use your GPU, set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    
    # Load inception model
    inception_model = torchvision.models.inception_v3(pretrained=True, transform_input=False).type(dtype)
    
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()
        
    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    #Compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py)) #cross-entropy is more stable than the original version with logs
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

        

if __name__ == '__main__':
    generated_imgs = []
    for filename in os.listdir(img_dir):
        if filename == ".DS_Store":
            continue
        img = misc.imread(img_dir + filename)
        generated_imgs.append(img)
        #if filename.endswith('.png'): #images must be pngs!
        #    img = misc.imread(img_dir + filename)
        #    generated_imgs.append(img)
    assert len(generated_imgs) > 0

    generated_imgs = np.asarray(generated_imgs)
    generated_imgs = generated_imgs.T[:][:][:][:3].T #way to get around weird problem with array slicing
    generated_imgs = np.moveaxis(generated_imgs,-1,1) #NxHxWxC -> NxCxHxW
    generated_imgs = 2 * (generated_imgs/np.amax(generated_imgs)) - 1 #normalize to between -1 and 1 as directed in the specs
    #^^ should normalization be per image?? ^^
    generated_imgs = torch.from_numpy(generated_imgs)
    #vrescaled = (v - minimum(v, axis=0)) * (newupper - newlower) / (max(v) - min(v)) + newlower

    print(inception_score(generated_imgs, cuda=False, batch_size=8, resize=True, splits=10))
