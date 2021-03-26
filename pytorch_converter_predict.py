'''Feature prediction: prediction (test) script'''


from __future__ import print_function

import glob
import os
from itertools import product
from time import time

import numpy as np
import scipy.io as sio
import hdf5storage

from fastl2lir import FastL2LiR

import bdpy
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.distcomp import DistComp

from bdpy.dataform import load_array, save_array

import torch


# Main #######################################################################

def main():
    # Read settings ----------------------------------------------------

    # Brain data
    brain_dir = '/home/share/data/fmri_shared/datasets/Deeprecon/fmriprep'
    subjects_list = {'TH' :  'TH_ImageNetTest_volume_native.h5'}

    rois_list = {
        'VC'  : 'ROI_VC = 1',
    }


    # Image features
    features_dir = '/home/ho/Documents/brain-decoding-examples/python/feature-prediction/data/features/ImageNetTest'
    network = 'caffe/VGG_ILSVRC_19_layers'
    features_list = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                     'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                     'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                     'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                     'fc6', 'fc7', 'fc8'][::-1]
    features_list = ['fc6', 'fc7', 'fc8'][::-1]
    target_subject = 'AM'

    Lambda = 0.1
    data_rep = 5

    # Model parameters
    gpu_device = 1

    # Results directory
    results_dir_root = './NCconverter_results'

    # Converter models
    nc_models_dir_root = os.path.join(results_dir_root, 'pytorch_converter_training', 'model')
    selected_converter_type = 'conv5'

    # Misc settings
    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

    # Pretrained model metadata
    pre_results_dir_root = '/home/share/data/contents_shared/ImageNetTraining/derivatives/feature_decoders'
    pre_analysis_basename = 'deeprecon_fmriprep_rep5_500voxel_allunits_fastl2lir_alpha100'
    pre_models_dir_root =os.path.join(pre_results_dir_root, pre_analysis_basename)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(os.path.join(brain_dir, dat_file))
                  for sbj, dat_file in subjects_list.items()}
    data_features = Features(os.path.join(features_dir, network))

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir_root)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi, feat in product(subjects_list, rois_list, features_list):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)


        # Distributed computation setup
        # -----------------------------
        subject_name = sbj+'2'+target_subject+'_'+str(data_rep*20)+'p'+'_lambda'+str(Lambda)
        analysis_id = analysis_basename + '-' + subject_name + '-' + roi + '-' + feat
        results_dir_prediction = os.path.join(results_dir_root, analysis_basename, 'decoded_features', network, feat, subject_name, roi)
        results_dir_accuracy = os.path.join(results_dir_root, analysis_basename, 'prediction_accuracy', network, feat, subject_name, roi)

        if os.path.exists(results_dir_prediction):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        if dist.islocked_lock():
            print('%s is already running. Skipped.' % analysis_id)
            continue



        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        x = data_brain[sbj].select(rois_list[roi])        # Brain data
        x_labels = data_brain[sbj].select('image_index')  # Image labels in the brain data

        # Target features and image labels (file names)
        y = data_features.get_features(feat)
        y_labels = data_features.index
        image_names = data_features.labels

        # Get test data
        x_test = x
        x_test_labels = x_labels

        y_test = y
        y_test_labels = y_labels

        # Averaging brain data
        x_test_labels_unique = np.unique(x_test_labels)
        x_test_averaged = np.vstack([np.mean(x_test[(x_test_labels == lb).flatten(), :], axis=0) for lb in x_test_labels_unique])

        print('Total elapsed time (data preparation): %f' % (time() - start_time))

        # Convert x_test_averaged
        nc_models_dir = os.path.join(nc_models_dir_root, subject_name, roi, 'model')
        x_test_averaged = test_ncconverter(nc_models_dir, x_test_averaged, gpu_device)

        # Prediction
        # ----------
        print('Prediction')

        start_time = time()
        y_pred = test_fastl2lir_div(os.path.join(pre_models_dir_root, network, feat, target_subject, roi, 'model'), x_test_averaged)
        print('Total elapsed time (prediction): %f' % (time() - start_time))

        # Calculate prediction accuracy
        # -----------------------------
        print('Prediction accuracy')

        start_time = time()

        y_pred_2d = y_pred.reshape([y_pred.shape[0], -1])
        y_true_2d = y.reshape([y.shape[0], -1])

        y_true_2d = get_refdata(y_true_2d, y_labels, x_test_labels_unique)

        n_units = y_true_2d.shape[1]

        accuracy = np.array([np.corrcoef(y_pred_2d[:, i].flatten(), y_true_2d[:, i].flatten())[0, 1]
                             for i in range(n_units)])
        accuracy = accuracy.reshape((1,) + y_pred.shape[1:])

        print('Mean prediction accuracy: {}'.format(np.mean(accuracy)))

        print('Total elapsed time (prediction accuracy): %f' % (time() - start_time))

        # Save results
        # ------------
        print('Saving results')

        makedir_ifnot(results_dir_prediction)
        makedir_ifnot(results_dir_accuracy)

        start_time = time()

        # Predicted features
        for i, lb in enumerate(x_test_labels_unique):
            # Predicted features
            feat = np.array([y_pred[i,]])  # To make feat shape 1 x M x N x ...

            image_filename = image_names[int(lb) - 1]  # Image labels are one-based image indexes

            # Save file name
            save_file = os.path.join(results_dir_prediction, '%s.mat' % image_filename)

            # Save
            hdf5storage.savemat(save_file, {u'feat' : feat},
                                format='7.3', oned_as='column', store_python_metadata=True)

        print('Saved %s' % results_dir_prediction)

        # Prediction accuracy
        save_file = os.path.join(results_dir_accuracy, 'accuracy.mat')
        hdf5storage.savemat(save_file, {u'accuracy' : accuracy},
                            format='7.3', oned_as='column', store_python_metadata=True)
        print('Saved %s' % save_file)

        print('Elapsed time (saving results): %f' % (time() - start_time))

        dist.unlock()

    print('%s finished.' % analysis_basename)


# Functions ##################################################################
def test_ncconverter(model_store, x, gpu_device=0):
    # Load NC converter
    print('Load NC converter')
    torch.cuda.set_device(gpu_device)
    print(model_store)   
    NCconverter = torch.load(os.path.join(model_store, 'NCconverter_L1.pt'))
    print(NCconverter)
    NCconverter.eval()

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(model_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(model_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)

    # Normalize X
    x = (x - x_mean) / x_norm
    converted_x = NCconverter(torch.from_numpy(x).float().to(gpu_device)).detach().cpu().numpy()
    converted_x = converted_x * y_norm + y_mean
    
    return converted_x



def test_fastl2lir_div(model_store, x, chunk_axis=1):
    # W: shape = (n_voxels, shape_features)
    if os.path.isdir(os.path.join(model_store, 'W')):
        W_files = sorted(glob.glob(os.path.join(model_store, 'W', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'W.mat')):
        W_files = [os.path.join(model_store, 'W.mat')]
    else:
        raise RuntimeError('W not found.')

    # b: shape = (1, shape_features)
    if os.path.isdir(os.path.join(model_store, 'b')):
        b_files = sorted(glob.glob(os.path.join(model_store, 'b', '*.mat')))
    elif os.path.isfile(os.path.join(model_store, 'b.mat')):
        b_files = [os.path.join(model_store, 'b.mat')]
    else:
        raise RuntimeError('b not found.')

    x_mean = hdf5storage.loadmat(os.path.join(model_store, 'x_mean.mat'))['x_mean']  # shape = (1, n_voxels)
    x_norm = hdf5storage.loadmat(os.path.join(model_store, 'x_norm.mat'))['x_norm']  # shape = (1, n_voxels)
    y_mean = hdf5storage.loadmat(os.path.join(model_store, 'y_mean.mat'))['y_mean']  # shape = (1, shape_features)
    y_norm = hdf5storage.loadmat(os.path.join(model_store, 'y_norm.mat'))['y_norm']  # shape = (1, shape_features)

    # Normalize X
    # This normalization is turned off in NCconveter prediction, because the output of NCconverter
    # already take this into account.
    # x = (x - x_mean) / x_norm

    # Prediction
    y_pred_list = []
    for i, (Wf, bf) in enumerate(zip(W_files, b_files)):
        print('Chunk %d' % i)

        start_time = time()
        W_tmp = load_array(Wf, key='W')
        b_tmp = load_array(bf, key='b')

        model = FastL2LiR(W=W_tmp, b=b_tmp)
        y_pred_tmp = model.predict(x)

        # Denormalize Y
        if y_mean.ndim == 2:
            y_pred_tmp = y_pred_tmp * y_norm + y_mean
        else:
            y_pred_tmp = y_pred_tmp * y_norm[:, [i], :] + y_mean[:, [i], :]

        y_pred_list.append(y_pred_tmp)

        print('Elapsed time: %f s' % (time() - start_time))

    return np.concatenate(y_pred_list, axis=chunk_axis)


# Classes ####################################################################

class Features(object):
    '''DNN features class.'''

    def __init__(self, dpath='./'):
        self.__dpath = dpath
        self.__c_feature_name = None
        self.__features = None
        self.labels = self.__get_labels()
        self.index = np.arange(len(self.labels)) + 1

    def __get_labels(self):
        labels = []

        first_layer_dir = sorted(os.listdir(self.__dpath))[0]
        dpath = os.path.join(self.__dpath, first_layer_dir)

        for fl in os.listdir(dpath):
            fpath = os.path.join(dpath, fl)

            if os.path.isdir(fpath):
                continue
            if os.path.splitext(fl)[1] != '.mat':
                continue
            labels.append(os.path.splitext(fl)[0])

        return sorted(labels)

    def get_features(self, layer):
        '''Return features in `layer`.'''

        if layer == self.__c_feature_name:
            return self.__features

        dpath = os.path.join(self.__dpath, layer)

        feat = []
        labels = []
        for fl in sorted(os.listdir(dpath)):
            fpath = os.path.join(dpath, fl)

            if os.path.isdir(fpath):
                continue
            if os.path.splitext(fl)[1] != '.mat':
                continue

            feat_tmp = sio.loadmat(fpath)['feat']
            feat.append(feat_tmp)
            labels.append(os.path.splitext(fl)[0])

        # Check label consistency
        if not np.array_equal(self.labels, labels):
            raise ValueError('Inconsistent labels.')

        self.__c_feature_name = layer
        self.__features = np.vstack(feat)

        return self.__features

# Pytorch setting ############################################################
class NCconverter_torch(torch.nn.Module): 
  
    def __init__(self, source_num, target_num): 
        super(NCconverter_torch, self).__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(source_num, 4096),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(4096, 1024),
                                              torch.nn.ReLU())
        self.decoder = torch.nn.Sequential(torch.nn.Linear(1024, 4096),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(4096, target_num))

  
    def forward(self, X): 

        return self.decoder(self.encoder(X))


# Entry point ################################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()
