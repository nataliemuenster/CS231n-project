

import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def show_images(images):
    print("original shape: ", images.shape)
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    print("reshaped: ", images.shape)
    plt.imshow(img)


def save_images(images, filename):
    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    plt.imsave(filename, images)

def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def count_params(model):
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count


# In[34]:

import os
from scipy import misc

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples



NOISE_DIM = 32 #96
batch_size = 32 #32

train_dir = './dataset/dataset/'
test_dir = './dataset/dataset/'
train_set = []
val_set = []

for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img = misc.imread(train_dir + filename)
        train_set.append(img)

for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        img = misc.imread(test_dir + filename)
        val_set.append(img)

train_set = torch.from_numpy(np.asarray(train_set))
val_set = torch.from_numpy(np.asarray(val_set))

print(train_set.shape)
print(val_set.shape)

NUM_TRAIN = train_set.shape[0] #512
NUM_VAL = val_set.shape[0] #128

train_set = train_set.transpose(1, 3)
test_set = val_set.transpose(1, 3)

#mnist_train = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True,
 #                          transform=T.ToTensor())
loader_train = DataLoader(torch.utils.data.TensorDataset(train_set), batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))

#mnist_val = dset.MNIST('./cs231n/datasets/MNIST_data', train=True, download=True,
 #                          transform=T.ToTensor())
loader_val = DataLoader(torch.utils.data.TensorDataset(train_set), batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, 0)) #NUM_VAL, NUM_TRAIN??

print(len(loader_train))
print(len(loader_val))
#elem is shape 1x32x3x28x28
#for i,elem in enumerate(loader_train):
#    if i < 2:
#        print(len(elem[0][0][0][0]))#[0][0][0][0]))#elem is shape: 2x128x1x28x28


"""
imgs = loader_train.__iter__().next()[0].view(batch_size, 784).numpy().squeeze()
show_images(imgs)
"""


# In[35]:

print(train_set.shape)
print(train_set.transpose(1, 3).shape)
train_set = train_set.transpose(1, 3)
test_set = test_set.transpose(1, 3)


# ## Random Noise
# Generate uniform noise from -1 to 1 with shape `[batch_size, dim]`.
# 
# Hint: use `torch.rand`.

# In[36]:

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.
    
    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    t = torch.Tensor(batch_size, dim).uniform_(-1, 1)
    return t


# Make sure noise is the correct shape and type:

# In[37]:

def test_sample_noise():
    batch_size = 3
    dim = 4
    torch.manual_seed(231)
    z = sample_noise(batch_size, dim)
    np_z = z.cpu().numpy()
    assert np_z.shape == (batch_size, dim)
    assert torch.is_tensor(z)
    assert np.all(np_z >= -1.0) and np.all(np_z <= 1.0)
    assert np.any(np_z < 0.0) and np.any(np_z > 0.0)
    print('All tests passed!')
    
test_sample_noise()


# In[38]:

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


# ## CPU / GPU
# By default all code will run on CPU. GPUs are not needed for this assignment, but will help you to train your models faster. If you do want to run the code on a GPU, then change the `dtype` variable in the following cell.

# In[39]:

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor ## UNCOMMENT THIS LINE IF YOU'RE ON A GPU!


# # Discriminator
# Our first step is to build a discriminator. Fill in the architecture as part of the `nn.Sequential` constructor in the function below. All fully connected layers should include bias terms. The architecture is:
#  * Fully connected layer with input size 784 and output size 256
#  * LeakyReLU with alpha 0.01
#  * Fully connected layer with input_size 256 and output size 256
#  * LeakyReLU with alpha 0.01
#  * Fully connected layer with input size 256 and output size 1
#  
# Recall that the Leaky ReLU nonlinearity computes $f(x) = \max(\alpha x, x)$ for some fixed constant $\alpha$; for the LeakyReLU nonlinearities in the architecture above we set $\alpha=0.01$.
#  
# The output of the discriminator should have shape `[batch_size, 1]`, and contain real numbers corresponding to the scores that each of the `batch_size` inputs is a real image.

# In[40]:

def discriminator():
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    model = nn.Sequential(
        Flatten(),
        nn.Linear(2352, 256), 
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(256, 256), 
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(256, 1)
    )
    return model

# # Generator
# Now to build the generator network:
#  * Fully connected layer from noise_dim to 1024
#  * `ReLU`
#  * Fully connected layer with size 1024 
#  * `ReLU`
#  * Fully connected layer with size 784
#  * `TanH` (to clip the image to be in the range of [-1,1])

def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch model implementing the architecture above.
    """
    
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024,2352), nn.Tanh()
    )
    return model


# # GAN Loss

def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


# In[45]:

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    true_labels = torch.ones(logits_real.size()).type(dtype)
    untrue_labels = torch.zeros(logits_fake.size()).type(dtype)
    
    true = bce_loss(logits_real, true_labels)
    generated = bce_loss(logits_fake, untrue_labels)
    
    return true+generated


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    true_labels = torch.ones(logits_fake.size()).type(dtype)
    loss = bce_loss(logits_fake, true_labels)#.mean()    
    return loss


# # Optimizing our loss
# Make a function that returns an `optim.Adam` optimizer for the given model with a 1e-3 learning rate, beta1=0.5, beta2=0.999. You'll use this to construct optimizers for the generators and discriminators for the rest of the notebook.

# In[48]:

def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.
    
    Input:
    - model: A PyTorch model that we want to optimize.
    
    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5,0.999))

    return optimizer


# # Training a GAN!
# 
# We provide you the main training loop... you won't need to change this function, but we encourage you to read through and understand it. 

# In[49]:

def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=20, 
              batch_size=128, noise_size=96, num_epochs=100):
    """
    Train a GAN!
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    #print(loader_train)
    iter_count = 0
    for epoch in range(num_epochs):
        for x in loader_train:
            x = x[0]
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.type(dtype)
            hi = 2* (real_data - 0.5)
            logits_real = D(2* (real_data - 0.5)).type(dtype)

            g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 3, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, NOISE_DIM).type(dtype)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 3, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count == 0 or iter_count % show_every == 0):
                imgs_numpy = fake_images.data.cpu().numpy()
                save_images(imgs_numpy, 'dataset/out/tmp_iter' + str(iter_count) + '.png')
#plt.show()
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
            iter_count += 1


# In[50]:

# Make the discriminator
D = discriminator().type(dtype)

# Make the generator
G = generator().type(dtype)

# Use the function you wrote earlier to get optimizers for the Discriminator and the Generator
D_solver = get_optimizer(D)
G_solver = get_optimizer(G)
# Run it!
run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, batch_size=batch_size, noise_size = NOISE_DIM)
