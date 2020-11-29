

#### [ We are still working on code release. This is not the final version yet. Let us know if you encounter any bugs, errors or have any questions (kangxue.yin@gmail.com)]


### Prerequisites

- Linux (tested under Ubuntu 16.04 )
- Python (tested under 3.6.2)
- TensorFlow (tested under 1.13.1-GPU )
- numpy, scipy, h5py, scipy, open3d, PyMCubes, tflearn, etc.

The code reuses some components from 
<a href="https://github.com/optas/latent_3d_points">latent_3d_points</a>,
<a href="https://github.com/charlesq34/pointnet2">pointnet2</a> 
and <a href="https://github.com/czq142857/IM-NET">IM-NET</a>.  Before run the code, please compile the customized TensorFlow operators under the folders "latent\_3d\_points/structural\_losses" and 
"pointnet\_plusplus/tf\_ops".

### Dataset

- Download the dataset and pretained models <a href="https://drive.google.com/u/0/uc?id=1htY0dARRDrOid4gjPHkWtzZNVvbP_rqo&export=download">HERE</a>.


### Usage

The commond lines for training and testing the models are all under the folder "./CMD_sh". You may need to open, read and modify the .sh files.

To train and test part alignment:
```
bash ./CMD_sh/partAlign_train_chair.sh
```

To train and test joint synthesis:
```
% first pretrain part encoders
bash ./CMD_sh/partAE_train_chair1234.sh

% then train the joint synthesis network and test it on input parts with GT joints
bash ./CMD_sh/jointSynthesis_train_chair.sh
```

To test joint synthesis for given parts from different objects:

First set ```diffShape="1"``` in ```"./CMD_sh/partAlign_train_chair.sh"```. And run it to export aligned parts randomly selected from different objects. 

Then run the test on the aligned parts:
```
bash ./CMD_sh/jointSynthesis_test_onRealOutput_chair.sh
```

### Poisson blending
Take a look at the folder "poisson-blending" 

### Point samping and data preprocessing
Take a look at the folder "data-preprocess" 

### Citation
If you find our work useful in your research, please consider citing:

    @inproceedings{yin2020coalesce,
        author = {Kangxue Yin, Zhiqin Chen, Siddhartha Chaudhuri, Matthew Fisher, Vladimir Kim and Hao Zhang}
        title = {COALESCE: Component Assembly by Learning to Synthesize Connections}
        booktitle = {Proc. of 3DV}
        year = {2020}
    }

