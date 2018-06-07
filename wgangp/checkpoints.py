import os
import torch

def save_checkpoint(model, directory, iteration):
  directory = os.path.join(directory, model.name)
  path = os.path.join(directory, str(iteration))
  if not os.path.exists(directory):
    os.makedirs(directory)
  
  torch.save(model.state_dict(), path)

def load_checkpoint(curr_model, directory, iteration):
  path = os.path.join(directory, curr_model.name, str(iteration))

  checkpoint = torch.load(path)
  curr_model.load_state_dict(checkpoint)

def save_images(vectors, tag):
  directory = './' + tag.replace(' ', '')
  if not os.path.exists(directory):
    os.mkdir(directory)
  it = 0
  for vec in vectors:
    torch.save(vec, './' + directory + '/image' + str(it) + '.jpg')
    it += 1
