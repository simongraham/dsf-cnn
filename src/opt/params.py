import tensorflow as tf
#### Training parameters

class_pcam = {
    'train_input_shape'  : [96, 96],
    'train_output_shape' : [ 1,  1],
    'infer_input_shape'  : [96, 96],
    'infer_output_shape' : [ 1,  1],
    'input_chans'  : 3,

    'label_names': ['Non-Tumour', 'Tumour'],

    'filter_sizes': [5, 7, 9],

    'input_norm': True,

    'training_phase'    : [
        {
            'nr_epochs': 50,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (5.0e-5, [('15', 1.0e-5), ('25', 1.0e-5), ('35', 1.0e-5), ('40', 2.0e-5), ('45', 1.0e-5)]), 
            },
            'pretrained_path'  : None, # randomly initialise weights
            'train_batch_size' : 32,
            'infer_batch_size' : 64,

            'model_flags' : {
                'freeze' : False
            }
        }
    ],
    'loss_term' : {'bce' : 1}, 

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_auc',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 64,
}
seg_gland = {
    'train_input_shape'  : [448, 448],
    'train_output_shape' : [448, 448],
    'infer_input_shape'  : [448, 448],
    'infer_output_shape' : [112, 112],
    'input_chans'  : 3,

    'filter_sizes': [5, 7, 11],

    'input_norm': True,

    'training_phase'    : [
        {
            'nr_epochs': 70,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-3, [('15', 1.0e-4), ('50', 5.0e-5)]), 
            },
            'pretrained_path'  : None, # randomly initialise weights
            'train_batch_size' : 6,
            'infer_batch_size' : 12,

            'model_flags' : {
                'freeze' : False
            }
        }
    ],
    'loss_term' : {'bce' : 1}, 

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 12,
}
seg_nuc = {
    'train_input_shape'  : [256, 256],
    'train_output_shape' : [256, 256],
    'infer_input_shape'  : [256, 256],
    'infer_output_shape' : [112, 112],
    'input_chans'  : 3,

    'filter_sizes': [5, 7, 11],

    'input_norm': True,

    'training_phase'    : [
        {
            'nr_epochs': 70,
            'manual_parameters' : { 
                # tuple(initial value, schedule)
                'learning_rate': (1.0e-3, [('15', 1.0e-4), ('30', 5.0e-5)]), 
            },
            'pretrained_path'  : None, # randomly initialise weights
            'train_batch_size' : 6,
            'infer_batch_size' : 12,

            'model_flags' : {
                'freeze' : False
            }
        }
    ],
    'loss_term' : {'bce' : 1, 'dice' : 1}, 

    'optimizer'           : tf.train.AdamOptimizer,

    'inf_auto_metric'   : 'valid_dice',
    'inf_auto_comparator' : '>',
    'inf_batch_size' : 12,
}
