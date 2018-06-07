from random import randint
from runner import run

for i in range(10):
  noise_dim = randint(50, 150)
  disc_channels = randint(32, 96)
  gen_channels = randint(32, 96)
  disc_iterations = randint(1, 8)
  learning_rate = 10**randint(-6, -2)
  tag = 'hyperparam-{}n-{}d-{}g-{}i-{}lr'.format(noise_dim, disc_channels, gen_channels, disc_iterations, learning_rate)
  print('Attempting {}'.format(tag))

  args = {
    'dataset': 'landmarks',
    'classes': [5554],
    'image_dim': 64,
    'use_cuda': True,
    'num_epochs': 6,
    'images_every': 250,
    'checkpoint_every': 500,
    'noise_dim': noise_dim,
    'disc_channels': disc_channels,
    'gen_channels': gen_channels,
    'disc_iterations': disc_iterations,
    'learning_rate': learning_rate,
    'tag': tag,
  }

  try:
    run(args)
  except:
    print('{} messed up :('.format(tag))
