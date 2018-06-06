import glob, os

from PIL import Image
from torch.utils.data import Dataset


class GoogleLandmark(Dataset):
  def __init__(self, root, classes=None, transform=None, preload=False):
    """ Intialize the Google Landmark dataset
    
    Args:
      - root: root directory of the dataset
      - classes: list of landmark ids to get subset of data from
      - tranform: a custom tranform function
      - preload: flag for preloading the dataset into memory
        (not recommended unless classes is not None)
    """
    self.images = None
    self.allowed_classes = classes
    self.labels = []
    self.filenames = []
    self.root = root
    self.transform = transform

    # read filenames
    if self.allowed_classes:
      for landmark_id in self.allowed_classes:
        filenames = glob.glob(os.path.join(root, '*-{}.jpg'.format(landmark_id)))
        for fn in filenames:
          label = os.path.splitext(fn)[0].split('-')[1]
          self.filenames.append(fn)
          self.labels.append(label)
    else:
      filenames = glob.glob(os.path.join(root, '*.jpg'))
      for fn in filenames:
        label = os.path.splitext(fn)[0].split('-')[1]
        self.filenames.append(fn)
        self.labels.append(label)
        
    # if preload dataset into memory
    if preload:
      self._preload()
      
    self.len = len(self.filenames)
                
  def _preload(self):
    """
    Preload dataset to memory
    """
    self.labels = []
    self.images = []
    for image_fn in self.filenames:            
      image = Image.open(image_fn)
      self.images.append(image.copy())
      image.close()

  def __getitem__(self, index):
    """ Get a sample from the dataset
    """
    if self.images is not None:
      # If dataset is preloaded
      image = self.images[index]
      label = self.labels[index]
    else:
      # If on-demand data loading
      image_fn = self.filenames[index]
      label = self.labels[index]
      image = Image.open(image_fn)
      
    # May use transform function to transform samples
    # e.g., random crop, whitening
    if self.transform is not None:
      image = self.transform(min(image.size))(image)
    # return image and label
    return image, label

  def __len__(self):
    """
    Total number of samples in the dataset
    """
    return self.len
