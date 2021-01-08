"""infer.py 

Main inference script.

Usage:
  infer.py [--gpu=<id>] [--mode=<mode>]
  infer.py (-h | --help)
  infer.py --version

Options:
  -h --help      Show this string.
  --version      Show version.
  --gpu=<id>     Comma separated GPU list. [default: 0]     
  --mode=<mode>  Inference mode- use either 'seg' or 'class'.
"""

from docopt import docopt
import argparse
import glob
import math
import os
import sys
from collections import deque
from keras.utils import HDF5Matrix

import cv2
import numpy as np

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from config import Config
from misc.utils import rm_n_mkdir,cropping_center

import json
import operator

from sklearn.metrics import roc_auc_score



def get_best_chkpts(path, metric_name, comparator='>'):
    """
    Return the best checkpoint according to some criteria.
    Note that it will only return valid path, so any checkpoint that has been
    removed wont be returned (i.e moving to next one that satisfies the criteria
    such as second best etc.)

    Args:
        path: directory contains all checkpoints, including the "stats.json" file
    """
    stat_file = path + '/stats.json'
    ops = {
        '>': operator.gt,
        '<': operator.lt,
    }

    op_func = ops[comparator]
    with open(stat_file) as f:
        info = json.load(f)

    if comparator == '>':
        best_value = -float("inf")
    else:
        best_value = +float("inf")

    best_chkpt = None
    for epoch_stat in info:
        epoch_value = epoch_stat[metric_name]
        if op_func(epoch_value, best_value):
            chkpt_path = "%s/model-%d.index" % (path,
                                                epoch_stat['global_step'])
            if os.path.isfile(chkpt_path):
                selected_stat = epoch_stat
                best_value = epoch_value
                best_chkpt = chkpt_path
    return best_chkpt, selected_stat
####


class InferClass(Config):

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back
            tissue: tissue mask created via otsu -> only process tissue regions
                    if it is provided
        """

        prob = predictor(x)[0]
        pred = np.argmax(prob, -1)
        pred = np.squeeze(pred)
        prob = np.squeeze(prob[..., 1])

        return prob, pred

    ####
    def run(self):
        if self.inf_auto_find_chkpt:
            print('-----Auto Selecting Checkpoint Basing On "%s" Through "%s" Comparison' %
                  (self.inf_auto_metric, self.inf_auto_comparator))
            model_path, stat = get_best_chkpts(
                self.save_dir, self.inf_auto_metric, self.inf_auto_comparator)
            print('Selecting: %s' % model_path)
            print('Having Following Statistics:')
            for key, value in stat.items():
                print('\t%s: %s' % (key, value))
        else:
            model_path = self.inf_model_path

        ####
        save_dir = self.inf_output_dir
        predict_list = [['case', 'prediction']]

        file_load_img = HDF5Matrix(
            self.inf_data_list[0] + 'camelyonpatch_level_2_split_test_x.h5', 'x')
        file_load_lab = HDF5Matrix(
            self.inf_data_list[0] + 'camelyonpatch_level_2_split_test_y.h5', 'y')

        true_list = []
        prob_list = []
        pred_list = []

        num_ims = file_load_img.shape[0]
        last_step = math.floor(num_ims / self.inf_batch_size)
        last_step = self.inf_batch_size * last_step
        last_batch = num_ims - last_step
        count = 0
        for start_batch in range(0, last_step+1, self.inf_batch_size):
            sys.stdout.write("\rProcessed (%d/%d)" % (start_batch, num_ims))
            sys.stdout.flush()
            if start_batch != last_step:
                img = file_load_img[start_batch:start_batch +
                                    self.inf_batch_size]
                img = img.astype('uint8')
                lab = np.squeeze(
                    file_load_lab[start_batch:start_batch+self.inf_batch_size])
            else:
                img = file_load_img[start_batch:start_batch+last_batch]
                img = img.astype('uint8')
                lab = np.squeeze(
                    file_load_lab[start_batch:start_batch+last_batch])

            prob, pred = self.__gen_prediction(img, predictor)

            for j in range(prob.shape[0]):
                predict_list.append([str(count), str(prob[j])])
                count += 1

            prob_list.extend(prob)
            pred_list.extend(pred)
            true_list.extend(lab)

        prob_list = np.array(prob_list)
        pred_list = np.array(pred_list)
        true_list = np.array(true_list)
        accuracy = (pred_list == true_list).sum() / np.size(true_list)
        error = (pred_list != true_list).sum() / np.size(true_list)

        print('Accurcy (%): ', 100*accuracy)
        print('Error (%): ', 100*error)
        if self.model_mode == 'class_pcam':
            auc = roc_auc_score(true_list, prob_list)
            print('AUC: ', auc)

        # Save predictions to csv
        rm_n_mkdir(save_dir)
        for result in predict_list:
            predict_file = open('%s/predict.csv' % save_dir, "a")
            predict_file.write(result[0])
            predict_file.write(',')
            predict_file.write(result[1])
            predict_file.write("\n")
            predict_file.close()
####


class InferSeg(Config):

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back
            tissue: tissue mask created via otsu -> only process tissue regions
                    if it is provided
        """
        step_size = self.infer_output_shape
        msk_size = self.infer_output_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = x.shape[0]
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        #### TODO: optimize this
        sub_patches = []
        skipped_idx = []
        # generating subpatches from orginal
        idx = 0
        for row in range(0, last_h, step_size[0]):
            for col in range(0, last_w, step_size[1]):
                win = x[row:row+win_size[0],
                        col:col+win_size[1]]
                sub_patches.append(win)
                idx += 1

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0]
            if win_size[0] > msk_size[0]:
                mini_output = cropping_center(mini_output, (diff_h, diff_w))
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            if win_size[0] > msk_size[0]:
                mini_output = cropping_center(mini_output, (diff_h, diff_w))
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape

        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(
            pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
            np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        # just crop back to original size
        pred_map = np.squeeze(pred_map[:im_h, :im_w])

        return pred_map

    ####
    def run(self, tta):

        if self.inf_auto_find_chkpt:
            print('-----Auto Selecting Checkpoint Basing On "%s" Through "%s" Comparison' %
                  (self.inf_auto_metric, self.inf_auto_comparator))
            model_path, stat = get_best_chkpts(
                self.save_dir, self.inf_auto_metric, self.inf_auto_comparator)
            print('Selecting: %s' % model_path)
            print('Having Following Statistics:')
            for key, value in stat.items():
                print('\t%s: %s' % (key, value))
        else:
            model_path = self.inf_model_path

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model=model_constructor(),
            session_init=get_model_loader(model_path),
            input_names=self.eval_inf_input_tensor_names,
            output_names=self.eval_inf_output_tensor_names,
            create_graph=False)
        predictor = OfflinePredictor(pred_config)

        for data_dir in self.inf_data_list:
            save_dir = self.inf_output_dir + '/raw/'
            file_list = glob.glob('%s/*%s' % (data_dir, self.inf_imgs_ext))
            file_list.sort()  # ensure same order

            rm_n_mkdir(save_dir)
            for filename in file_list:
                start = time.time()
                filename = os.path.basename(filename)
                basename = filename.split('.')[0]
                print(data_dir, basename, end=' ', flush=True)

                ##
                img = cv2.imread(data_dir + filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pred_map = self.__gen_prediction(img, predictor)

                np.save('%s/%s.npy' % (save_dir, basename), [pred_map])
                end = time.time()
                diff = str(round(end-start, 2))
                print('FINISH. TIME: %s' % diff)


####
if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    if args['--gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']
        nr_gpus = len(args['--gpu'].split(','))
    
    if args['--mode'] is None:
        raise Exception(
            'Mode cannot be empty. Use either "class" or "seg".')

    if args['--mode'] == 'class':
        infer = InferClass()
    elif args['--mode'] == 'seg':
        infer = InferSeg()
    else:
        raise Exception(
            'Mode not recognised. Use either "class" or "seg".')

    infer.run()
