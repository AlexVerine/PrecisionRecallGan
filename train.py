""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import time
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F 
from torch.nn import Parameter as P
import torchvision
import logging
# Import my stuff
import inception_utils
import precision_recall_kyn_utils
import precision_recall_simon_utils

import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  # By default, skip init if resuming training.
  if config['resume']:
    logging.info('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  # Prepare root folders if necessary
  utils.prepare_root(config)
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  config['experiment_name'] = experiment_name
  print(experiment_name)



  utils.setup_logging(config)
  logging.info('Experiment name is %s' % experiment_name)
  config['resume'] += config['resume_no_optim']

  # Next, build the model
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)
  
   # If using EMA, prepare it
  if config['ema']:
    logging.info('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    logging.info('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    logging.info('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D)
  logging.info(G)
  logging.info(D)
  logging.info('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'best_P':0, 'best_R':0,'best_P+R':0, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    logging.info('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'] if config['load_weights'] else None,
                       G_ema if config['ema'] else None, 
                      load_optim=not config['resume_no_optim'])
    logging.info(f"Resume training with lrD: {D.optim.param_groups[0]['lr']:e} and lrG: {G.optim.param_groups[0]['lr']:e}")

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
                                            experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  logging.info('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  logging.info('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  if config['use_multiepoch_sampler']:
    size_loader = int(np.ceil(loaders[0].sampler.num_samples/D_batch_size))
  else:
    size_loader = len(loaders[0])
  config['total_itr'] = int((size_loader)*(config['num_epochs']-state_dict['epoch']))

  if (size_loader//5)<500:
    config['log_itr'] = [i*(size_loader//5) for i in range(5)]+[size_loader-1]
  else:
    config['log_itr'] =  [size_loader-1]
  # Prepare inception metrics: FID and IS
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
  #Prepare vgg metrics: Precision and Recall
  get_pr_metric = precision_recall_kyn_utils.prepare_pr_metrics(config)
  # get_pr_curve = precision_recall_simon_utils.prepare_pr_curve(config)
  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':
    train = train_fns.GAN_training_function(G, D, GD, z_, y_, 
                                            ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample,
                              G=(G_ema if config['ema'] and config['use_ema']
                                 else G),
                              z_=z_, y_=y_, config=config)

  logging.info(f'Beginning training at epoch {state_dict["epoch"]} for {config["total_itr"]} iterations.')
  t_init = time.time()
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own
    t0 = time.time()
    for i, (x, y) in enumerate(loaders[0]):
      if i % size_loader  == 0 :
        t0 = time.time()

      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      metrics = train(x, y, train_G=(i>30 or not config['resume_no_optim']))
      train_log.log(itr=int(state_dict['itr']), **metrics)
      
      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if i in config["log_itr"] or i%250 == 0:
        e = 1+ i//size_loader if config['use_multiepoch_sampler'] else epoch

        logging.info(f'[{e:d}/{config["num_epochs"]:d}]({i+1}/{size_loader})({int(time.time()-t0):d}s/{int((size_loader -i%size_loader)*(time.time()-t0)/(i%size_loader+1)):d}s) : {state_dict["itr"] }     '+ 'Mem used (Go) {:.2f}/{:.2f}'.format(torch.cuda.mem_get_info(0)[1]/1024**3
                       -torch.cuda.mem_get_info(0)[0]/1024**3, torch.cuda.mem_get_info(0)[1]/1024**3))

        logging.info('\t'+', '.join(['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]))
          # logging.info()
          # logging.info(', '.join(['itr: %d' % state_dict['itr']] 
          #                  + ['%s : %+4.3f' % (key, metrics[key])
          #                  for key in metrics]))
      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
  #
        if config['G_eval_mode']:
          logging.info('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if not (state_dict['itr'] % config['test_every']):
        if config['G_eval_mode']:
          logging.info('Switchin G to eval mode...')
          G.eval()
        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, get_pr_metric, experiment_name, test_log)
        logging.info(f'\tEstimated time: {(time.time()-t_init)*config["total_itr"]/state_dict["itr"] // 86400:.0f} days and '
              + f'{ ( ( time.time()-t_init)*config["total_itr"]/state_dict["itr"] % 86400) / 3600:2.1f} hours.')
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1
  
  if config['G_eval_mode']:
    logging.info('Switchin G to eval mode...')
    G.eval()
    if config['ema']:
      G_ema.eval()
  train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                            state_dict, config, experiment_name)

def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  config['mode'] = 'train'

  if config['data_root'] is None:
      config['data_root'] = os.environ.get('DATADIR', None)
  if config['data_root'] is None:
      ValueError("the following arguments are required: --data_dir")
  if config['data_root'] is None:
    config['data_root'] = os.environ.get('DATADIR', None)


  print(config)
  run(config)

if __name__ == '__main__':
  main()
