import importlib
import numpy as np
import tensorflow as tf

import opt.augs as augs

from model.utils.gconv_utils import get_basis_filters, get_rot_info, get_basis_params

#### 
class Config(object):
    def __init__(self, ):

        self.seed = 10

        self.model_mode = 'seg_nuc' # choose seg_gland, seg_nuc, class_pcam
        self.filter_type = 'steerable' # choose steerable or standard

        self.nr_orients = 8 # number of orientations for the filters

        #### Dynamically setting the hyper-param and aug files into variable
        param_file = importlib.import_module('opt.params')
        param_dict = param_file.__getattribute__(self.model_mode)

        for variable, value in param_dict.items():
            self.__setattr__(variable, value)

        self.data_ext = '.npy' 
        # list of directories containing training and validation files. Each directory contains one numpy file per image.
        # if self.model_mode = 'seg_nuc' or 'seg_gland', data is of size [H,W,4] (RGB + instance label)
        # # if self.mode == 'class_pcam', data is of size [(H*W*C)+1], where the final value is the class label
        self.train_dir = ['/media/simon/Storage 1/Data/Nuclei/patches/kumar/train/']
        self.valid_dir = ['/media/simon/Storage 1/Data/Nuclei/patches/kumar/valid/']

        # nr of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4 

        exp_id = 'v1.0'
        # loading chkpts in tensorflow, the path must not contain extra '/'
        self.log_path = '/media/simon/Storage 1/dsf-cnn/checkpoints/' # log root path
        self.save_dir = '%s/%s/%s_%s_%s' % (self.log_path, self.model_mode, self.filter_type, self.nr_orients, exp_id) # log file destination

        #### Info for running inference
        self.inf_auto_find_chkpt = False
        # path to checkpoints will be used for inference, replace accordingly
        self.inf_model_path  = self.save_dir + 'model-xxxx.index'

        # paths to files for inference. Note, for PCam we use the original .h5 file.
        self.inf_imgs_ext = '.tif'
        self.inf_data_list = [
            '/media/simon/Storage 1/Data/Nuclei/kumar/test/Images/'
        ]

        output_root = '/media/simon/Storage 1/output/'
        self.inf_output_dir = '%s/%s_%s/' % (output_root, self.filter_type, self.nr_orients) # log file destination

        if self.filter_type == 'steerable':
            # Generate the basis filters- only need to do this once before training
            self.basis_filter_list = []
            self.rot_matrix_list = []
            for ksize in self.filter_sizes:
                alpha_list, beta_list, bl_list = get_basis_params(ksize)
                b_filters, freq_filters = get_basis_filters(alpha_list, beta_list, bl_list, ksize)
                self.basis_filter_list.append(b_filters)
                self.rot_matrix_list.append(get_rot_info(self.nr_orients, freq_filters))
    
        # for inference during evalutaion mode i.e run by inferer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

    def get_model(self):
        model_constructor = importlib.import_module('model.%s.graph' % self.model_mode)
        model_constructor = model_constructor.Graph       
        return model_constructor 
    
     # refer to https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html for 
    # information on how to modify the augmentation parameters. Pipeline can be modified in opt/augs.py

    def get_train_augmentors(self, input_shape, output_shape, view=False):
        return augs.get_train_augmentors(self, input_shape, output_shape, view)
        
    def get_valid_augmentors(self, input_shape, output_shape, view=False):
        return augs.get_valid_augmentors(self, input_shape, output_shape, view)
    
