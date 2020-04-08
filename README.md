# Dense Steerable Filter CNNs for Expoiting Rotational Symmetry in Histology Images

A densely connected rotation-equivariant CNN for histology image analysis. <br />

[Link]() to the paper. <br />

## Repository Structure 

- `src/` contains executable files used to run the model. Further information on running the code can be found in the corresponding directory.
- `loader/`contains scripts for data loading and self implemented augmentation functions.
- `misc/`contains util scripts. 
- `model/class_pcam/` model architecture for dsf-cnn on pcam dataset 
- `model/seg_nuc/` model architecture for dsf-cnn on kumar dataset 
- `model/seg_gland/` model architecture for dsf-cnn on crag dataset 
- `model/utils/` contains util scripts for the models. 
- `opt/` contains scripts that define the model hyperparameters and augmentation pipeline. 
- `config.py` is the configuration file. Paths need to be changed accordingly.
- `train.py` and `infer.py` are the training and inference scripts respectively.
- `process.py` is the post processing script for obtaining the final instances for segmentation. 
- `extract_patches.py` is the patch extraction script. 

<p float="left">
  <img src="/feature_maps.gif" alt="Segmentation" width="400" />
  <img src="/feature_maps.gif" alt="Segmentation" width="400" />
</p>

## Citation 

If any part of this code is used, please give appropriate citation to our paper. <br>



