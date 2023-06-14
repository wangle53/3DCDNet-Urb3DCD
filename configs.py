#!/usr/bin/python
# -*- coding: UTF-8 -*-
import ml_collections as mlc
import numpy as np
import os

def train_cfg():
    
    """returns training configuration."""
    
    cfg = mlc.ConfigDict()
    cfg.resume = False
    cfg.display = True
    cfg.print_rate = 20
    cfg.batch_size = 2
    cfg.epoch = 40
    
    
    # network setting
    cfg.use_rgb = False
    if cfg.use_rgb:
        cfg.in_dim = 6
    else:
        cfg.in_dim = 3
    cfg.out_dim = 64
    cfg.sub_sampling_ratio = [2, 2, 2, 2]
    cfg.down_rate = np.prod(cfg.sub_sampling_ratio)
    cfg.num_layers = len(cfg.sub_sampling_ratio)
    cfg.k_neighbors = 16 # The k value in LFA module
    
    
    # dataset setting
    cfg.n_samples = 1024
    cfg.vx = 50 # crop pc with resolution of [50, 50]
    cfg.vy = 50 
    cfg.norm_data = True
    cfg.sub_datasets = ['1-Lidar05', '2-Lidar10', '3-Lidar05Noisy', '4-Photogrametry', '5-MultiSensor']
    cfg.sub_dataset = '1-Lidar05' # change to different sub_datasets
    cfg.train_flag = 'TrainLarge-1c' # change to different training datasets listed in '1-Lidar05'
    cfg.val_flag = 'Val'
    cfg.test_flag = 'Test'
    
    
    # path
    cfg.path = mlc.ConfigDict()
    cfg.Urb3DCD_path = 'F:/Urb3DCD/IEEE_Dataset_V1' # change to your path
        # prepare data setting
    cfg.if_prepare_data = True   # if pre-process dataset to accelerate training phase.
    cfg.save_dataset_path = 'F:/Urb3DCD' # change to your path
    cfg.path.dataset_root = os.path.join(cfg.save_dataset_path, 'subPCs_vx{}_vy{}'.format(cfg.vx, cfg.vy))
    cfg.path.prepare_data = os.path.join(cfg.save_dataset_path, 
                            'prapared_data_vx{}_vy{}_r{}_n{}_k{}'.format(cfg.vx, 
                                                                     cfg.vy, 
                                                                     str(cfg.sub_sampling_ratio[0]), 
                                                                     str(cfg.n_samples), 
                                                                     str(cfg.k_neighbors))) 
    cfg.path.train_dataset_path = os.path.join(cfg.path.dataset_root, cfg.sub_dataset, cfg.train_flag)
    cfg.path.test_dataset_path = os.path.join(cfg.path.dataset_root, cfg.sub_dataset, cfg.test_flag)
    cfg.path.val_dataset_path = os.path.join(cfg.path.dataset_root, cfg.sub_dataset, cfg.val_flag)
    
    
    # '.txt' path
    cfg.path.txt_path = './data_{}_{}'.format(cfg.vx, cfg.vy)
    cfg.path.train_txt = cfg.path.txt_path + '/{}/{}.txt'.format(cfg.sub_dataset, cfg.train_flag)
    cfg.path.val_txt = cfg.path.txt_path + '/{}/{}.txt'.format(cfg.sub_dataset, cfg.val_flag)
    cfg.path.test_txt = cfg.path.txt_path + '/{}/{}.txt'.format(cfg.sub_dataset, cfg.test_flag)
    
    # output path
    cfg.path.outputs = './outputs_{}_{}'.format(cfg.sub_dataset, cfg.train_flag)
    cfg.path.weights_save_dir = cfg.path.outputs + '/weights'
    cfg.path.best_weights_save_dir = cfg.path.outputs + '/best_weights'
    cfg.path.val_prediction = cfg.path.outputs + '/val_prediction'
    cfg.path.test_prediction = cfg.path.outputs + '/test_prediction'
    cfg.path.test_prediction_PCs = cfg.path.outputs + '/test_prediction_PCs'
    cfg.path.feature = cfg.path.outputs + '/feature'
    
    # optimizer setting
    cfg.optimizer = mlc.ConfigDict()
    cfg.optimizer.lr = 0.001
    cfg.optimizer.momentum = 0.9
    cfg.optimizer.weight_decay = 0.0005
    cfg.optimizer.lr_step_size = 1
    cfg.optimizer.gamma = 0.95
    
    # validation and testing setting
    cfg.save_prediction = True
    cfg.criterion = 'miou'  # criterion for selecting models: 'miou' or 'oa'
    
    return cfg

CONFIGS = {
    'Train': train_cfg(),
    }