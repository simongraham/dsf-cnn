"""train.py 

Main training script.

Usage:
  train.py [--gpu=<id>] [--view=<dset>]
  train.py (-h | --help)
  train.py --version

Options:
  -h --help      Show this string.
  --version      Show version.
  --gpu=<id>     Comma separated GPU list.  
  --view=<dset>  View dataset- use either 'train' or 'valid'.
"""

from docopt import docopt
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorpack import Inferencer, logger
from tensorpack.callbacks import (
    DataParallelInferenceRunner,
    ModelSaver,
    MinSaver,
    MaxSaver,
    ScheduledHyperParamSetter,
)
from tensorpack.tfutils import SaverRestore, get_model_loader
from tensorpack.train import (
    SyncMultiGPUTrainerParameterServer,
    TrainConfig,
    launch_train_with_config,
)

import loader.loader as loader
from config import Config
from misc.utils import get_files

from sklearn.metrics import roc_auc_score


class StatCollector(Inferencer, Config):
    """
    Accumulate output of inference during training.
    After the inference finishes, calculate the statistics
    """

    def __init__(self, prefix="valid"):
        super(StatCollector, self).__init__()
        self.prefix = prefix

    def _get_fetches(self):
        return self.train_inf_output_tensor_names

    def _before_inference(self):
        self.true_list = []
        self.pred_list = []

    def _on_fetches(self, outputs):
        pred, true = outputs
        self.true_list.extend(true)
        self.pred_list.extend(pred)

    def _after_inference(self):
        # ! factor this out
        def _dice(true, pred, label):
            true = np.array(true[..., label], np.int32)
            pred = np.array(pred[..., label], np.int32)
            inter = (pred * true).sum()
            total = (pred + true).sum()
            return 2 * inter / (total + 1.0e-8)

        stat_dict = {}
        pred = np.array(self.pred_list)
        true = np.array(self.true_list)

        if self.model_mode == "seg_gland":
            # Get the segmentation stats

            pred = pred[..., :2]
            true = true[..., :2]

            # Binarize the prediction
            pred[pred > 0.5] = 1.0

            stat_dict[self.prefix + "_dice_obj"] = _dice(true, pred, 0)
            stat_dict[self.prefix + "_dice_cnt"] = _dice(true, pred, 1)

        elif self.model_mode == "seg_nuc":
            # Get the segmentation stats

            pred = pred[..., :3]
            true = true[..., :3]

            # Binarize the prediction
            pred[pred > 0.5] = 1.0

            stat_dict[self.prefix + "_dice_np"] = _dice(true, pred, 0)
            stat_dict[self.prefix + "_dice_mk_blb"] = _dice(true, pred, 1)
            stat_dict[self.prefix + "_dice_mk_cnt"] = _dice(true, pred, 2)

        else:
            # Get the classification stats

            # Convert vector to scalar prediction
            prob = np.squeeze(pred[..., 1])
            pred = np.argmax(pred, -1)
            pred = np.squeeze(pred)
            true = np.squeeze(true)

            accuracy = (pred == true).sum() / np.size(true)
            error = (pred != true).sum() / np.size(true)

            stat_dict[self.prefix + "_acc"] = accuracy * 100
            stat_dict[self.prefix + "_error"] = error * 100

            if self.model_mode == "class_pcam":
                auc = roc_auc_score(true, prob)
                stat_dict[self.prefix + "_auc"] = auc

        return stat_dict


###########################################


class Trainer(Config):
    def get_datagen(self, batch_size, mode="train", view=False):
        if mode == "train":
            augmentors = self.get_train_augmentors(
                self.train_input_shape, self.train_output_shape, view
            )
            data_files = get_files(self.train_dir, self.data_ext)
            # Different data generators for segmentation and classification
            if self.model_mode == "seg_gland" or self.model_mode == "seg_nuc":
                data_generator = loader.train_generator_seg
            else:
                data_generator = loader.train_generator_class
            nr_procs = self.nr_procs_train
        else:
            augmentors = self.get_valid_augmentors(
                self.train_input_shape, self.train_output_shape, view
            )
            # Different data generators for segmentation and classification
            data_files = get_files(self.valid_dir, self.data_ext)
            if self.model_mode == "seg_gland" or self.model_mode == "seg_nuc":
                data_generator = loader.valid_generator_seg
            else:
                data_generator = loader.valid_generator_class
            nr_procs = self.nr_procs_valid

        # set nr_proc=1 for viewing to ensure clean ctrl-z
        nr_procs = 1 if view else nr_procs
        dataset = loader.DatasetSerial(data_files)
        if self.model_mode == "seg_gland" or self.model_mode == "seg_nuc":
            datagen = data_generator(
                dataset,
                shape_aug=augmentors[0],
                input_aug=augmentors[1],
                label_aug=augmentors[2],
                batch_size=batch_size,
                nr_procs=nr_procs,
            )
        else:
            datagen = data_generator(
                dataset,
                shape_aug=augmentors[0],
                input_aug=augmentors[1],
                batch_size=batch_size,
                nr_procs=nr_procs,
            )

        return datagen

    def view_dataset(self, mode="train"):
        assert mode == "train" or mode == "valid", "Invalid view mode"
        if self.model_mode == "seg_gland" or self.model_mode == "seg_nuc":
            datagen = self.get_datagen(4, mode=mode, view=True)
            loader.visualize(datagen, 4)
        else:
            # visualise more for classification- don't need to show label
            datagen = self.get_datagen(8, mode=mode, view=True)
            loader.visualize(datagen, 8)
        return

    def run_once(self, opt, sess_init=None, save_dir=None):

        train_datagen = self.get_datagen(opt["train_batch_size"], mode="train")
        valid_datagen = self.get_datagen(opt["infer_batch_size"], mode="valid")

        ###### must be called before ModelSaver
        if save_dir is None:
            logger.set_logger_dir(self.save_dir)
        else:
            logger.set_logger_dir(save_dir)

        ######
        model_flags = opt["model_flags"]
        model = self.get_model()(**model_flags)
        ######
        callbacks = [
            ModelSaver(max_to_keep=1, keep_checkpoint_every_n_hours=None),
        ]

        for param_name, param_info in opt["manual_parameters"].items():
            model.add_manual_variable(param_name, param_info[0])
            callbacks.append(ScheduledHyperParamSetter(param_name, param_info[1]))
        # multi-GPU inference (with mandatory queue prefetch)
        infs = [StatCollector()]
        callbacks.append(
            DataParallelInferenceRunner(valid_datagen, infs, list(range(nr_gpus)))
        )
        if self.model_mode == "seg_gland":
            callbacks.append(MaxSaver("valid_dice_obj"))
        elif self.model_mode == "seg_nuc":
            callbacks.append(MaxSaver("valid_dice_np"))
        else:
            callbacks.append(MaxSaver("valid_auc"))

        steps_per_epoch = train_datagen.size() // nr_gpus

        config = TrainConfig(
            model=model,
            callbacks=callbacks,
            dataflow=train_datagen,
            steps_per_epoch=steps_per_epoch,
            max_epoch=opt["nr_epochs"],
        )
        config.session_init = sess_init

        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpus))
        tf.reset_default_graph()  # remove the entire graph in case of multiple runs
        return

    def run(self):
        def get_last_chkpt_path(prev_phase_dir):
            stat_file_path = prev_phase_dir + "/stats.json"
            with open(stat_file_path) as stat_file:
                info = json.load(stat_file)
            chkpt_list = [epoch_stat["global_step"] for epoch_stat in info]
            last_chkpts_path = "%smodel-%d.index" % (prev_phase_dir, max(chkpt_list))
            return last_chkpts_path

        phase_opts = self.training_phase

        if len(phase_opts) > 1:
            for idx, opt in enumerate(phase_opts):

                log_dir = "%s/%02d" % (self.save_dir, idx)
                if opt["pretrained_path"] == -1:
                    pretrained_path = get_last_chkpt_path(prev_log_dir)
                    init_weights = SaverRestore(
                        pretrained_path, ignore=["learning_rate"]
                    )
                elif opt["pretrained_path"] is not None:
                    init_weights = get_model_loader(pretrained_path)
                self.run_once(opt, sess_init=init_weights, save_dir=log_dir + "/")
                prev_log_dir = log_dir
        else:

            opt = phase_opts[0]
            if "pretrained_path" in opt:
                if opt["pretrained_path"] == None:
                    init_weights = None
                elif opt["pretrained_path"] == -1:
                    log_dir_prev = "%s" % self.save_dir
                    pretrained_path = get_last_chkpt_path(log_dir_prev)
                    init_weights = SaverRestore(
                        pretrained_path, ignore=["learning_rate"]
                    )
                else:
                    init_weights = get_model_loader(opt["pretrained_path"])
            self.run_once(opt, sess_init=init_weights, save_dir=self.save_dir)

        return


###########################################################################


if __name__ == "__main__":

    args = docopt(__doc__)
    print(args)

    trainer = Trainer()

    if args["--view"] and args["--gpu"]:
        raise Exception("Supply only one of --view and --gpu.")

    if args["--view"]:
        if args["--view"] != "train" and args["--view"] != "valid":
            raise Exception('Use "train" or "valid" for --view.')
        trainer.view_dataset(args["--view"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]
        nr_gpus = len(args["--gpu"].split(","))
        trainer.run()
