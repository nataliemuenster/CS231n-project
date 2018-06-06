import argparse, utils

import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset

from models.wgangp import WGAN_GP
from defaults import get_defaults
from train import train

def run(user_args):
  # Get full set of user and default args.
  args = get_defaults()
  args_len = len(args)
  args.update(user_args)
  assert(len(args) == args_len)  # Ensure there's no typos in the argparser below.
  print("Running with arguments:")
  for arg, value in args.items():
    print("{}: {}".format(arg, value))

  # Setup CPU or GPU environment
  if args['use_cuda'] and not torch.cuda.is_available():
    args['use_cuda'] = False
    print("Could not access CUDA. Using CPU...")
  elif args['use_cuda']:
    print("Using CUDA.")
    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor
  else:      
    device = torch.device('cpu')
    dtype = torch.FloatTensor

  # Define the WGAN
  W = WGAN_GP(
    dtype,
    noise_dim=args['noise_dim'],
    image_dim=args['image_dim'],
    image_channels=args['image_channels'],
    disc_channels=args['disc_channels'],
    gen_channels=args['gen_channels'],
    cuda=args['use_cuda']
  )

  # Initialize weights and parameters
  if args['init_method'] == 'gaussian':
    utils.gaussian_initialize(W, 0.02)
  elif args['init_method'] == 'xavier':
    utils.xavier_initialize(W)

  # Prepare and load the data
  mnist_train = dset.MNIST('./data/MNIST', train=True, download=True,
                 transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.ToPILImage(),
                  transforms.Pad(2),
                  transforms.ToTensor(),
                ]))
  loader_train = DataLoader(mnist_train, batch_size=args['batch_size'],
                shuffle=True, drop_last=True, pin_memory=True if args['use_cuda'] else False)
  # imgs = loader_train.__iter__().next()[0].view(batch_size, 1 * 32 * 32).numpy().squeeze()
  # utils.show_images(imgs)

  # Train!
  train(W, loader_train, dtype, args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run WGAN-GP on the dataset.')

  # Model parameters
  parser.add_argument('--noise_dim', type=int, help='Dimension of sample noise for generator.')
  parser.add_argument('--image_dim', type=int, help='Dimension of input images.')
  parser.add_argument('--image_channels', type=int, help='Number of channels in input images.')
  parser.add_argument('--disc_channels', type=int, help='Number of channels for discriminator layers.')
  parser.add_argument('--gen_channels', type=int, help='Number of channels for generator layers.')

  # Train-time dynamics
  parser.add_argument('--batch_size', type=int, help='Number of images per batch.')
  parser.add_argument('--num_epochs', type=int, help='Number of epochs to train for.')
  parser.add_argument('--init_method', type=str, choices=['gaussian', 'xavier', 'none'], help='Weight initialization method to use.')
  parser.add_argument('--disc_iterations', type=int, help='Number of iterations of discriminator per iteration of generator.')
  parser.add_argument('--disc_warmup_length', type=int, help='Number of iterations before discriminator initially exits rapid train mode.')
  parser.add_argument('--disc_warmup_iterations', type=int, help='Number of iterations in a discriminator rapid train cycle.')
  parser.add_argument('--disc_rapid_train_interval', type=int, help='Frequency of discriminator rapid train cycle during training.')
  parser.add_argument('--lambda_val', type=float, help='Hyperparameter for gradient penalty.')

  # Optimizer hyperparameters
  parser.add_argument('--learning_rate', type=float, help='Learning rate for ADAM optimizer.')
  parser.add_argument('--beta1', type=float, help='Beta 1 value for ADAM optimizer.')
  parser.add_argument('--beta2', type=float, help='Beta 2 value for ADAM optimizer.')
  parser.add_argument('--weight_decay', type=float, help='Weight decay for ADAM optimizer.')

  # Reporter configuration
  parser.add_argument('--images_every', type=int, help='How often (in iterations) to report images during training.')
  parser.add_argument('--losses_every', type=int, help='How often (in iterations) to report losses during training.')
  parser.add_argument('--sample_size', type=int, help='Number of images per report during during training.')

  # Cuda
  parser.add_argument('--use_cuda', action='store_true', help='Flag for using Cuda GPU instead of CPU.')
  
  args = vars(parser.parse_args())
  args = {k : v for k, v in args.items() if v is not None}
  run(args)

  