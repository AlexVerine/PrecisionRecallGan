''' Datasets
    This file contains definitions for our CIFAR, ImageFolder, and HDF5 datasets
'''
import os
import os.path
import sys
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
import logging
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.datasets.utils import download_url, check_integrity
import torch.utils.data as data
from torch.utils.data import DataLoader
         
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
  images = []
  dir = os.path.expanduser(dir)
  for target in tqdm(sorted(os.listdir(dir))):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in sorted(fnames):
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

  return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)


class ImageFolder(data.Dataset):
  """A generic data loader where the images are arranged in this way: ::

      root/dogball/xxx.png
      root/dogball/xxy.png
      root/dogball/xxz.png

      root/cat/123.png
      root/cat/nsdf3.png
      root/cat/asd932_.png

  Args:
      root (string): Root directory path.
      transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
      loader (callable, optional): A function to load an image given its path.

   Attributes:
      classes (list): List of the class names.
      class_to_idx (dict): Dict with items (class_name, class_index).
      imgs (list): List of (image path, class_index) tuples
  """

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, load_in_mem=False, 
               index_filename='imagenet_imgs.npz', **kwargs):
    classes, class_to_idx = find_classes(root)
    # Load pre-computed image directory walk
    if os.path.exists(index_filename):
      logging.info('Loading pre-saved Index file %s...' % index_filename)
      imgs = np.load(index_filename)['imgs']
    # If first time, walk the folder directory and save the 
    # results to a pre-computed file.
    else:
      logging.info('Generating  Index file %s...' % index_filename)
      imgs = make_dataset(root, class_to_idx)
      np.savez_compressed(index_filename, **{'imgs' : imgs})
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                           "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader
    self.load_in_mem = load_in_mem
    
    if self.load_in_mem:
      logging.info('Loading all images into memory...')
      self.data, self.labels = [], []
      for index in tqdm(range(len(self.imgs))):
        path, target = imgs[index][0], imgs[index][1]
        self.data.append(self.transform(self.loader(path)))
        self.labels.append(target)
          

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    if self.load_in_mem:
        img = self.data[index]
        target = self.labels[index]
    else:
      path, target = self.imgs[index]
      img = self.loader(str(path))
      if self.transform is not None:
        img = self.transform(img)
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    # logging.info(img.size(), target)
    return img, int(target)

  def __len__(self):
    return len(self.imgs)

  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    tmp = '    Transforms (if any): '
    fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    tmp = '    Target Transforms (if any): '
    fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
    return fmt_str
        

''' ILSVRC_HDF5: A dataset to support I/O from an HDF5 to avoid
    having to load individual images all the time. '''
import h5py as h5
import torch
class ILSVRC_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies
      
    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])
    
    # self.transform = transform
    self.target_transform = target_transform   
    
    # Set the transform here
    self.transform = transform
    
    # load the entire dataset into memory? 
    self.load_in_mem = load_in_mem
    
    # If loading into memory, do so now
    if self.load_in_mem:
      logging.info('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]
    
    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]
    
   
    # if self.transform is not None:
        # img = self.transform(img)
    # Apply my own transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    return img, int(target)

  def __len__(self):
      return self.num_imgs
      # return len(self.f['imgs'])

class CelebA_HDF5(data.Dataset):
  def __init__(self, root, transform=None, target_transform=None,
               load_in_mem=False, train=True,download=False, validate_seed=0,
               val_split=0, **kwargs): # last four are dummies
      
    self.root = root
    self.num_imgs = len(h5.File(root, 'r')['labels'])
    
    # self.transform = transform
    self.target_transform = target_transform   
    
    # Set the transform here
    self.transform = transform
    
    # load the entire dataset into memory? 
    self.load_in_mem = load_in_mem
    
    # If loading into memory, do so now
    if self.load_in_mem:
      logging.info('Loading %s into memory...' % root)
      with h5.File(root,'r') as f:
        self.data = f['imgs'][:]
        self.labels = f['labels'][:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is class_index of the target class.
    """
    # If loaded the entire dataset in RAM, get image from memory
    if self.load_in_mem:
      img = self.data[index]
      target = self.labels[index]
    
    # Else load it from disk
    else:
      with h5.File(self.root,'r') as f:
        img = f['imgs'][index]
        target = f['labels'][index]
    
   
    # if self.transform is not None:
        # img = self.transform(img)
    # Apply my own transform
    img = ((torch.from_numpy(img).float() / 255) - 0.5) * 2
    
    if self.target_transform is not None:
      target = self.target_transform(target)
    
    return img, int(target)

  def __len__(self):
      return self.num_imgs
      # return len(self.f['imgs'])

import pickle
class CIFAR10(dset.CIFAR10):

  def __init__(self, root, train=True,
           transform=None, target_transform=None,
           download=True, validate_seed=0,
           val_split=0, load_in_mem=True, **kwargs):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.train = train  # training set or test set
    self.val_split = val_split

    if download:
      self.download()

    if not self._check_integrity():
      raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    # now load the picked numpy arrays    
    self.data = []
    self.labels= []
    for fentry in self.train_list:
      f = fentry[0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data.append(entry['data'])
      if 'labels' in entry:
        self.labels += entry['labels']
      else:
        self.labels += entry['fine_labels']
      fo.close()
        
    self.data = np.concatenate(self.data)
    # Randomly select indices for validation
    if self.val_split > 0:
      label_indices = [[] for _ in range(max(self.labels)+1)]
      for i,l in enumerate(self.labels):
        label_indices[l] += [i]  
      label_indices = np.asarray(label_indices)
      
      # randomly grab 500 elements of each class
      np.random.seed(validate_seed)
      self.val_indices = []           
      for l_i in label_indices:
        self.val_indices += list(l_i[np.random.choice(len(l_i), int(len(self.data) * val_split) // (max(self.labels) + 1) ,replace=False)])
    
    if self.train=='validate':    
      self.data = self.data[self.val_indices]
      self.labels = list(np.asarray(self.labels)[self.val_indices])
      
      self.data = self.data.reshape((int(50e3 * self.val_split), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    
    elif self.train:
      logging.info(np.shape(self.data))
      if self.val_split > 0:
        self.data = np.delete(self.data,self.val_indices,axis=0)
        self.labels = list(np.delete(np.asarray(self.labels),self.val_indices,axis=0))
          
      self.data = self.data.reshape((int(50e3 * (1.-self.val_split)), 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      if sys.version_info[0] == 2:
        entry = pickle.load(fo)
      else:
        entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()
      self.data = self.data.reshape((10000, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      
  def __getitem__(self, index):
    """
    Args:
        index (int): Index
    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target
      
  def __len__(self):
      return len(self.data)


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]



class CelebA(data.Dataset):
  """
  The CelebA dataset of
  (Liu et al., 2015): https://arxiv.org/abs/1411.7766
  preprocessed to 64x64 as in
  (Larsen et al. 2016): https://arxiv.org/abs/1512.09300
  (Dinh et al., 2017): https://arxiv.org/abs/1605.08803

  From https://github.com/laurent-dinh/models/blob/master/real_nvp/celeba_formatting.py:
  "
  Download img_align_celeba.zip from
  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html under the
  link "Align&Cropped Images" in the "Img" directory and list_eval_partition.txt
  under the link "Train/Val/Test Partitions" in the "Eval" directory.Then do:
  unzip img_align_celeba.zip
  "

  Subsequently, move the files img_align_celeba.zip and list_eval_partition.txt
  into folder data/celeba/raw/
  """
  def __init__(self, root, transform=None, load_in_mem=True, **kwargs):
    super().__init__()
    split='train'
    assert split in {'train','valid','test'}
    self.raw_folder = 'raw'
    self.processed_folder = 'processed'
    self.root = root
    self.split = split
    self.transform = transform
    
    if not self._check_raw():
      raise RuntimeError('Dataset not found.\n\nFrom docstring:\n\n' + self.__doc__)

    if not self._check_processed():
      logging.info('here')
      self.process()
      
      
    if self.split=='train':
      self.data = pd.read_csv(os.path.join(self.processed_data_folder, 'train.txt'), header=None)
      self.img_dir = self.processed_train_folder
    elif self.split=='valid':
      self.data = pd.read_csv(os.path.join(self.processed_data_folder, 'valid.txt'), header=None)
      self.img_dir = self.processed_valid_folder
    elif self.split=='test':
      self.data = pd.read_csv(os.path.join(self.processed_data_folder, 'test.txt'), header=None)
      self.img_dir = self.processed_test_folder


  def __getitem__(self, idx):
    """
    Args:
    index (int): Index
    Returns:
    tensor: image
    """

    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
    img = Image.open(img_path)
    if self.transform is not None:
      img = self.transform(img)
    return img, 0

  def __len__(self):
    return len(self.data)

  @property
  def raw_data_folder(self):
    return os.path.join(self.root, self.raw_folder)

  @property
  def raw_zip_file(self):
    return os.path.join(self.raw_data_folder, 'img_align_celeba.zip')

  @property
  def raw_txt_file(self):
    return os.path.join(self.raw_data_folder, 'list_eval_partition.txt')

  def _check_raw(self):
    return os.path.exists(self.raw_zip_file) and os.path.exists(self.raw_txt_file)

  @property
  def processed_data_folder(self):
    return os.path.join(self.root, self.processed_folder)

  @property
  def processed_train_folder(self):
    return os.path.join(self.processed_data_folder, 'train')

  @property
  def processed_valid_folder(self):
    return os.path.join(self.processed_data_folder, 'valid')

  @property
  def processed_test_folder(self):
    return os.path.join(self.processed_data_folder, 'test')

  def _check_processed(self):
    return os.path.exists(self.processed_train_folder) and os.path.exists(self.processed_valid_folder) and os.path.exists(self.processed_test_folder)

  def process_file_list(self, zipfile_object, file_list, processed_filename):
        
    with open(processed_filename+'.txt', 'w') as f:
      for i, jpg_file in enumerate(file_list):
  
        if (i+1)%1000 == 0:
          logging.info(f'File  {i+1}/{len(file_list)}\r')
  
        ## Read file
        img_bytes = zipfile_object.read('img_align_celeba/' + jpg_file)
        img = Image.open(io.BytesIO(img_bytes))
  
        ## Crop image
        # Coordinates taken from Line 981 in
        # https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_multiscale_dataset.py
        # Coordinates of upper left corner: (40, 15)
        # Size of cropped image: (148, 148)
        cropped_img = crop(img, 40, 15, 148, 148)
        
        ## Resize image
        # Resizing taken from Line 995-996 in
        # https://github.com/laurent-dinh/models/blob/master/real_nvp/real_nvp_multiscale_dataset.py
        resized_img = resize(img, size=(64,64), interpolation=Image.BILINEAR)
        
        name = f'image{i}.jpg'
        f.write(name+'\n')
        resized_img.save(processed_filename+'/'+name, quality=100)


  def process(self):

    logging.info("Reading filenames...")
    train_files = []
    valid_files = []
    test_files = []
    for line in open(self.raw_txt_file, 'r'):
      a, b = line.split()
      if b=='0':
        train_files.append(a)
      elif b=='1':
        valid_files.append(a)
      elif b=='2':
        test_files.append(a)

    logging.info("Reading zip file...")
    zip = zipfile.ZipFile(self.raw_zip_file, 'r')

    if not os.path.exists(self.processed_data_folder):
      os.mkdir(self.processed_data_folder)
      os.mkdir(self.processed_train_folder)
      os.mkdir(self.processed_valid_folder)
      os.mkdir(self.processed_test_folder)

    logging.info("Preparing training data...")
    self.process_file_list(zipfile_object=zip, file_list=train_files, processed_filename=self.processed_train_folder)

    logging.info("Preparing validation data...")
    self.process_file_list(zipfile_object=zip, file_list=valid_files, processed_filename=self.processed_valid_folder)

    logging.info("Preparing test data...")
    self.process_file_list(zipfile_object=zip, file_list=test_files, processed_filename=self.processed_test_folder)

    zip.close()




class FFHQ(data.Dataset):
  """ Flickr-Faces-HQ (FFHQ)
  https://github.com/NVlabs/ffhq-dataset
  """
  def __init__(self, root, transform=None, load_in_mem=True, **kwargs):
    super(FFHQ, self).__init__()
    self.root = root
    self.image_list = self._get_image_list()
    self.transform = transform

  def _get_image_list(self):
    image_list = ['{:05d}.png'.format(i) for i in range(70000)]
    return image_list

  def __len__(self):
      return len(self.image_list)

  def __getitem__(self, index):
    image = Image.open(os.path.join(self.root, self.image_list[index]))
    if self.transform is not None:
      image = self.transform(image)
    return image, 0


