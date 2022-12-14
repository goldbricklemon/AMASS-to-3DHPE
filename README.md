# AMASS-to-3DHPE

Conversion of the [AMASS](https://github.com/nghorbani/amass) dataset to a pure joint-based, Human3.6M compatible 3D human pose dataset.

This conversion script was written for the WACV '23 paper [Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers](https://arxiv.org/abs/2210.06110).

If you are interested in this work, take a look at the official repository on [GitHub](https://github.com/goldbricklemon/uplift-upsample-3dhpe).

## Updates
* 2022-12-14: Added extended README and installation instructions.


## Installation
This conversion tool requires the installation of the AMASS toolchain, including PyTorch and Human Body Prior. 
Follow the [installation instructions](https://github.com/nghorbani/amass#installation) from AMASS.

### Body Models
AMASS uses the [SMPL+H body model](https://mano.is.tue.mpg.de/). 
Follow the AMASS instructions and place the SMPL+H body model and [DMPLs](https://smpl.is.tue.mpg.de/) in `support_data/body_models/` directory:

```
support_data
└── body_models
    ├── dmpls/
    │   ├── female/
    │   │   └── model.npz
    │   ├── male
    │   │   └── model.npz
    │   └── neutral
    │       └── model.npz
    └── smplh/
        ├── female/
        │   └── model.npz
        ├── male
        │   └── model.npz
        └── neutral
            └── model.npz
```

### Human3.6m Joint Regressors
The standard 17-point skeleton from Human3.6m does not coincide with the default SMPL/SMPL+H joints. 
We therefore use a custom optimized joint regressor to obtain 17 compatible 3D joints from the SMPL vertices.
Download the joint regressor from the [SPIN](https://github.com/nkolot/SPIN/tree/master/data) repository using its `fetch_data.sh` script 
and place the regressor `J_regressor_h36m.npy` under `support_data/`.

### Custom Regressors
This conversion script is not limited to the Human3.6M joint regressor. 
You can provide your own regressor and convert the SMPL+H meshes to a different body model (i.e. other number or definition of joints).
The regressor is specified as a (N x 6890) matrix, where N is the number of regressed joints. 
Each joint is represented as a linear combination (typically sparse) of the 6890 SMPL+H vertices.
Save your custom regressor as a serialized float32 numpy array (i.e. with `np.save()`).

## AMASS Dataset
Next, [download](https://amass.is.tue.mpg.de/download.php) the AMASS dataset. 
Since AMASS is a collection of many different motion capture datasets, you can download your own preferred subset of datasets.

**NOTE**: This script only supports datasets that have been SMPLified with the SMPL+H model. 
Only download datasets as `SMPL+H G`. Some datasets are not available in this format.

Download each dataset, extract its contents and place them in a directory with the name of that dataset. 
The following lists all datasets (and the name of the corresponding directories) that have been tested so far:

  * CMU
  * DanceDB
  * MPILimits
  * TotalCapture
  * EyesJapanDataset
  * HUMAN4D
  * KIT
  * BMLhandball
  * BMLmovi
  * BMLrub
  * EKUT
  * TCDhandMocap
  * ACCAD
  * Transitionsmocap
  * MPIHDM05
  * SFU
  * MPImosh

Place all those directories (or the subset you want) in `data/`

## Conversion

We provide an example for Human3.6M-compatible conversion with the specified joint regressor from above. 
Each individual dataset is converted into a dictionary hierarchy of subjects and sequences:

```
{
  "[SUBJECT 1]": 
     {
       "[SEQUENCE 1]":
         {
           "positions_3d": np.ndarray, shape (#frames, N, 3),
           "frame_rate": float
         },
         "[SEQUENCE 2]":
         {
          ...
         }
     },
   "[SUBJECT 2]":
     ...
}
```
Here, `#frames` is the number of frames of each mocap sequence, 
`N` is the number of regressed joints (17 in case of Human3.6M) and 
`frame_rate` is the frame rate of the mocap sequence.

Since the mocap data in AMASS has a frame rate of 60Hz or more, but Human3.6M is in 50Hz, 
we resample the mocap sequences to this target frame rate.

The dictionary is then saved as a compressed numpy archive `"[DATASET_NAME].npz"` in a specified output directory `[OUTPUT_DIR]`.

You can run the conversion with the following conversion script from the `src` directory:
```
python convert.py 
--amass_dir ../data
--out_dir [OUTPUT_DIR]
--joint_regressor ../support_data/J_regressor_h36m.npy
--r 50
--gpu_id 0
--batch_size 2048
```

Here, we run the rigging and skinning of the SMPL meshes on the GPU, with a batch size of 2048. 
You can reduce the batch size or omit the gpu switch entirely (CPU mode) if needed.
See also
```
python convert.py --help
```

## Citation

In case this work is useful for your research, please consider citing:

    @InProceedings{einfalt_up3dhpe_WACV23,
    title={Uplift and Upsample: Efficient 3D Human Pose Estimation with Uplifting Transformers},
    author={Einfalt, Moritz and Ludwig, Katja and Lienhart, Rainer},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2023},
    }

## Acknowledgments

Our code is heavily influenced by the demo code from AMASS:

* [AMASS@GitHub](https://github.com/nghorbani/amass)
