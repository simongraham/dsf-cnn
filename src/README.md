# Training and Inference Instructions

## Choosing the network

The model to use and the selection of other hyperparameters is selected in `config.py`. The models available are:
- Classification PCam DSF-CNN: `model/class_pcam/graph.py`
- Segmentation CRAG DSF-CNN: `model/seg_gland/graph.py`
- Segmentation Kumar DSF-CNN: `model/seg_nuc/graph.py`

## Modifying Hyperparameters

To modify hyperparameters, refer to `opt/params.py` 

## Augmentation

To modify the augmentation pipeline, refer to `get_train_augmentors()` in `opt/augs.py`. Refer to [this webpage](https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html) for information on how to modify the augmentation parameters.

## Data Format

For segmentation, store patches in a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. For classification, save as a 1D array of size [(H * W * C)+1], where H, W and C refer to height, width and number of channels. The final value is the label of the image (starting from 0). 

## Training

To train the network, the command is: <br/>

`python train.py --gpu='<gpu_ids>'` <br/>
where gpu_id denotes which GPU will be used for training. For example, if we are using GPU number 0 and 1, the command is: <br/>
`python train.py --gpu='0,1'` <br/>

Before training, set in `config.py`:
- path to the data directories
- path where checkpoints will be saved
- path to where the output will be saved

## Inference

To generate the network predictions, the command is: <br/>
`python infer.py --gpu='<gpu_id>' --mode='<mode>` <br/>
Currently, the inference code only supports 1 GPU. For `'<mode>`, use `'seg'` or `'class'`. Use `'class'` when processing PCam and `'seg'` when processing Kumar and CRAG.   

Before running inference, set in `config.py`:
- path where the output will be saved
- path to data root directories
- path to model checkpoint. 

## Post Processing 

To obtain the final segmentation, use the command: <br/>
`python process.py <br/>
for post-processing the network predictions. Note, this is only for segmentation networks.

