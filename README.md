# MSSL
This is a TensorFlow implementation of MSSL as described in the following paper: 

``Xiaoli Liu, Jianqin Yin, MSSL: Multi-scale Semi-decoupled Spatiotemporal Learning for 3D human motion prediction, 2021.''
The paper is corresponding to the paper in ArXiv as follows:
Liu X, Yin J. SDMTL: Semi-Decoupled Multi-grained Trajectory Learning for 3D human motion prediction[J]. arXiv preprint arXiv:2010.05133, 2020.

## Setup
Required python libraries: tensorflow (>=1.0) + opencv + numpy.
Tested in ubuntu+  GTX 1080Ti with cuda (>=8.0) and cudnn (>=5.0).

## Datasets
Human3.6M, CMU-Mocap, 3DPW.
```
the processed data file will be available at: https://pan.baidu.com/s/1iVsvRC_PUeteY3Oi50teHA （password：123a）
```

## Training/Testing
Use the `scripts/h36m/train_MSSL_h36m_long_term.sh` or `scripts/h36m/train_MSSL_h36m_short_term.sh` script to train/test the model on Human3.6M dataset for short-term or long-term predictions by the following commands:
```shell
cd scripts/h36m
sh train_MSSL_h36m_short_term.sh  # for short-term prediction on Human3.6M
sh train_MSSL_h36m_long_term # for long-term predictions on Human3.6M
```
You might want to change folders in `scripts` to train on CMU-Mocap or 3DPW datasets.


## Citation
If you use this code for your research, please consider citing:
```latex
@article{liu2020sdmtl,
  title={SDMTL: Semi-Decoupled Multi-grained Trajectory Learning for 3D human motion prediction},
  author={Liu, Xiaoli and Yin, Jianqin},
  journal={arXiv preprint arXiv:2010.05133},
  year={2020}
}
```

## Contact
A part of code adopts from PredCNN at https://github.com/xzr12/PredCNN.git. 

