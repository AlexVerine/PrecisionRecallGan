''' Calculate VGG dataset Features
 This script iterates over the dataset and calculates the latent 
 activations of the VGG net (needed for Precision/Recalll).
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 

import utils
import precision_recall_kyn_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser

def prepare_parser():
  usage = 'Calculate and store vgg features.'
  parser = ArgumentParser(description=usage)
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)') 
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_pr_images', type=int, default=10000,
    help='Default number of image to compute  (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser

def run(config):
  # Get loader
  config['drop_last'] = False
  loaders = utils.get_data_loaders(**config)

  device = 'cuda'
  total = 0
  imgs = [] 
  with torch.no_grad():
    for i, (x, _) in enumerate(tqdm(loaders[0])):
      print(f'ap {torch.mean(x):.2f}, {torch.min(x):.2f}, {torch.max(x):.2f}')

      for j in range(x.size(0)):
        if total >= config['num_pr_images']:
          continue
        imgs.append(x[j:j+1, :, :, :])
        total +=1
    print(imgs[0].shape)
    print(f'Evaluating {len(imgs)} images with vgg.')
    ipr = precision_recall_kyn_utils.IPR(batch_size=64, k=3, num_samples=config['num_pr_images'], model=None)

    ipr.compute_manifold_ref(imgs)
  ipr.save_ref('samples/features/'+config['dataset'].strip('_hdf5')+'_vgg_features.npz')


def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()
