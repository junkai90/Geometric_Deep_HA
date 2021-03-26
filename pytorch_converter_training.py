'''Feature prediction: decoders training script'''


#from __future__ import print_function

import os
from itertools import product
from datetime import datetime
from time import time, sleep
import csv
import warnings
import glob

import hdf5storage
import numpy as np

import bdpy
from bdpy.util import makedir_ifnot, get_refdata, divide_chunks
from bdpy.distcomp import DistComp
from bdpy.dataform import load_array

import torch

import json
import sys

from training import train
from preprocess import MeshData, GraphData

# Main #######################################################################

def main():
    # Data settings ----------------------------------------------------

    # Brain data
    brain_dir = '/home/share/data/fmri_shared/datasets/Deeprecon/fmriprep'
    subjects_list = {'TH' :  'TH_ImageNetTraining_volume_native.h5'}
    target_data = {'AM': 'AM_ImageNetTraining_volume_native.h5'}

    rois_list = {
        'LH'  : 'VertexLeft',
    }


    # Assume the pretrained model is trained with TH data.
    # We want to train NCconverter using ES data.
    target_subject = 'AM'

    # Model parameters
    data_rep = 5

    # Model parameters
    lr_rate = 0.01
    epoch = 200
    batch = 1
    gpu_device = 0

    # Results directory
    results_dir_root = './NCconverter_results'

    # geometry dir
    geometry_dir = './surf'

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]



    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for sbj, dat_file in subjects_list.items()}

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir_root)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')


    for sbj, roi in product(subjects_list, rois_list):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('Target subject: %s' % target_subject)
        print('ROI:        %s' % roi)


        # Setup
        # -----
        subject_name = sbj+'2'+target_subject+'_'+str(data_rep*20)+'p'
        analysis_id = analysis_basename + '-' + subject_name + '-' + roi 
        results_dir = os.path.join(results_dir_root, analysis_basename, 'model', subject_name, roi, 'model')
        makedir_ifnot(results_dir)


        # Check whether the analysis has been done or not.
        check_file = os.path.join(results_dir, analysis_id + '.done')
        if os.path.exists(check_file):
            print('%s is already done and skipped' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # geometry data
        geofiles = (os.path.join(geometry_dir, '{}.white'.format(roi.lower())),
                    os.path.join(geometry_dir, '{}.pial'.format(roi.lower())))
        mesh = MeshData(geofiles)
        edges, pseudo = mesh.edge_pseudo()

        # Brain data
        x = data_brain[sbj].select(rois_list[roi])        # Brain data
        x_labels = data_brain[sbj].select('image_index')  # Image labels in the brain data

        target_brain_data = bdpy.BData(os.path.join(brain_dir, target_data[target_subject]))
        y = target_brain_data.select(rois_list[roi])
        y_labels = target_brain_data.select('image_index')

        # Get training data
        x_train = x
        x_train_labels = x_labels

        y_train = y
        y_train_labels = y_labels
        del x, y, x_labels, y_labels

        print('Total elapsed time (data preparation): %f' % (time() - start_time))

        # Model training
        # --------------
        print('Model training')
        start_time = time()
        train_NCconverter(x_train, y_train,
                          x_train_labels, y_train_labels,
                          edges, pseudo,
                          lr_rate=lr_rate, batch=batch,
                          output=results_dir, save_chunk=True,
                          axis_chunk=1, tmp_dir='tmp',
                          comp_id=analysis_id,
                          gpu_device=gpu_device, epoch=epoch)
        print('Total elapsed time (model training): %f' % (time() - start_time))

    print('%s finished.' % analysis_basename)


# Functions ##################################################################

def train_NCconverter(x, y, x_labels, y_labels, edges, pseudo,
                      lr_rate=0.01, batch=64,
                      output='./NCconverter_results.mat', save_chunk=False,
                      axis_chunk=1, tmp_dir='./tmp',
                      comp_id=None, gpu_device=0, epoch=500):

    makedir_ifnot(output)
    makedir_ifnot(tmp_dir)

    if y.ndim == 4:
        # The Y input to the NCconveter has to be strictly number of samples x number of features
        y = y.reshape((y.shape[0], -1))
    elif y.ndim == 2:
        pass
    else:
        raise ValueError('Unsupported feature array shape')

    # Preprocessing ----------------------------------------------------------
    print('Preprocessing')
    start_time = time()

    # Normalize X (fMRI data)
    x_mean = np.mean(x, axis=0)[np.newaxis, :] # np.newaxis was added to match Matlab outputs
    x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]
    x_normalized = (x - x_mean) / x_norm

    # Normalize Y (DNN features)
    y_mean = np.mean(y, axis=0)[np.newaxis, :]
    y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]
    y_normalized = (y - y_mean) / y_norm



    print('Elapsed time: %f' % (time() - start_time))

    # Model training loop ----------------------------------------------------


    comp_id_t = comp_id + 'NCconverter'
    results_dir = os.path.join(output)
    result_model = os.path.join(results_dir, 'NCconverter.pt')


    makedir_ifnot(results_dir)

    if os.path.exists(result_model):
        print('%s already exists and skipped' % result_model)
        return


    dist = DistComp(lockdir=tmp_dir, comp_id=comp_id_t)
    if dist.islocked():
        print('%s is already running. Skipped.' % comp_id_t)
        return


    dist.lock()

    start_time = time()


    print('Training')

    # add bias term in X
    #x_normalized = np.concatenate([x_normalized, np.ones((x_normalized.shape[0],1))], axis=1)


    # Align Y to X labels
    x_index = np.argsort(x_labels.flatten())
    x_labels_aligned = x_labels[x_index]

    y_index = np.argsort(y_labels.flatten())
    y_labels_aligned = y_labels[y_index]

    #y_index = np.array([np.where(y_labels == xl)[0] for xl in x_labels]).flatten()
    #y_aligned = y_normalized[y_index, :]
    #y_labels_aligned = y_labels[y_index]
    x_aligned = x_normalized[x_index,:]
    y_aligned = y_normalized[y_index,:]
    print(x_labels_aligned[:20])
    print(y_labels_aligned[:20])
    # np.random.seed(88)
    # x_aligned = np.random.permutation(x_aligned)
    # np.random.seed(88)
    # y_aligned = np.random.permutation(y_aligned)

    # Data
    graph = GraphData(x_aligned, y_aligned, edges, pseudo)
    graph_dat_list = graph.data

    # Model training
    #torch.cuda.set_device(gpu_device)
    #model = NCconverter_torch(x_aligned.shape[1], y_aligned.shape[1])
    model = train(graph_dat_list, lr_rate=lr_rate, epoch=epoch, batch=batch)

    # Save chunk results
    torch.save(model, result_model)
    print('Saved %s' % result_model)


    del(y_aligned)
    etime = time() - start_time
    print('Elapsed time: %f' % etime)
    dist.unlock()

    del(x_normalized)

    # Save results -----------------------------------------------------------


    print('Saving normalization parameters.')
    norm_param = {'x_mean' : x_mean, 'y_mean' : y_mean,
                    'x_norm' : x_norm, 'y_norm' : y_norm}
    save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
    for sv in save_targets:
        save_file = os.path.join(results_dir, sv + '.mat')
        if not os.path.exists(save_file):
            hdf5storage.savemat(save_file, {sv: norm_param[sv]},
                                format='7.3', oned_as='column', store_python_metadata=True)
            print('Saved %s' % save_file)

    if not save_chunk:
        # Merge results into 'model'mat'
        raise NotImplementedError('Result merging is not implemented yet.')

    return None
       

# Pytorch setting ############################################################
# class dataset(Dataset):
#     def __init__(self, X, Y):
#         self.Y = Y
#         self.X = X

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, index):
#         return self.X[index], self.Y[index]

# class NCconverter_torch(torch.nn.Module): 
  
#     def __init__(self, source_num, target_num): 
#         super(NCconverter_torch, self).__init__()
#         self.encoder = torch.nn.Sequential(torch.nn.Linear(source_num, 4096),
#                                               torch.nn.ReLU(),
#                                               torch.nn.Linear(4096, 1024),
#                                               torch.nn.ReLU())
#         self.decoder = torch.nn.Sequential(torch.nn.Linear(1024, 4096),
#                                            torch.nn.ReLU(),
#                                            torch.nn.Linear(4096, target_num))

  
#     def forward(self, X): 

#         return self.decoder(self.encoder(X))

    # def NCconvert(self, X):
    #     return self.linear(X)


# Entry point ################################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
