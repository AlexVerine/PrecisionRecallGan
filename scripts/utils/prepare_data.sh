#!/bin/bash
python make_hdf5.py --dataset C10 --batch_size 256  --data_dir ~/data
python calculate_inception_moments.py --dataset C10 --data_root ~/data --batch_size 256	
python calculate_vgg_features.py --dataset C10 --data_root ~/data
python calculate_inception_features.py --dataset C10 --data_root ~/data
