import argparse, utils

import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import reporter
from models.wgangp import WGAN_GP
from defaults import get_defaults
from train import train
from datamanager import GoogleLandmark

def run(user_args):
  # Get full set of user and default args.
  args = get_defaults()
  args_len = len(args)
  args.update(user_args)
  assert(len(args) == args_len + 1)  # Ensure there's no typos in the argparser below.
  print("Running with arguments:")
  for arg, value in args.items():
    print("{}: {}".format(arg, value))
  print()

  # Setup CPU or GPU environment
  if args['use_cuda'] and not torch.cuda.is_available():
    print("Could not access CUDA. Using CPU...")
    args['use_cuda'] = False
    device = torch.device('cpu')
    dtype = torch.FloatTensor
  elif args['use_cuda']:
    print("Using CUDA.")
    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor
  else:
    print("Using CPU.")
    device = torch.device('cpu')
    dtype = torch.FloatTensor

  print("Initializing model...")
  # Define the WGAN
  W = WGAN_GP(
    dtype,
    noise_dim=args['noise_dim'],
    image_dim=args['image_dim'],
    image_channels=args['image_channels'],
    disc_channels=args['disc_channels'],
    gen_channels=args['gen_channels'],
    dataset_name=args['dataset'],
    cuda=args['use_cuda']
  )

  # Initialize weights and parameters
  if args['init_method'] == 'gaussian':
    utils.gaussian_initialize(W, 0.02)
  elif args['init_method'] == 'xavier':
    utils.xavier_initialize(W)

  print("Initializing dataset...")
  # Prepare and load the data
  if args['dataset'] == 'mnist':
    dataset = dset.MNIST('./data/MNIST', train=True, download=True,
                   transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.Pad(2),
                    transforms.ToTensor(),
                  ]))
  else:
    dataset = GoogleLandmark('./data/train', [5554],
                   transform=lambda c: transforms.Compose([
                    transforms.CenterCrop(c),
                    transforms.Resize(args['image_dim']),
                    transforms.ToTensor(),
                  ]))
  data_loader = DataLoader(dataset, batch_size=args['batch_size'],
                shuffle=True, drop_last=True, pin_memory=True if args['use_cuda'] else False)

  # Visualize training images in Visdom.
  reporter.visualize_images(
    data_loader.__iter__().next()[0],
    'Training data samples',
    env=W.name
  )

  # Train!
  print("Training model...")
  train(W, data_loader, dtype, args)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run WGAN-GP on a dataset.')

  parser.add_argument('dataset', type=str, choices=['mnist', 'landmarks'], help='Dataset to run WGAN-GP on.')

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

  