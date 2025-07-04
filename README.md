# RoSD_SAR  [image text](https://github.com/douling843/RoSD_SAR/blob/main/images/linux-original.svg){width=80px}



![image text](https://github.com/douling843/RoSD_SAR/blob/main/images/fig3.jpg)

 This repository includes the official implementation of the paper:  

👋   **RoSD-SAR: Robust Ship Detection in SAR Images with Noisy Box Labels**

👨‍💻   **Code:** [GitHub](https://github.com/douling843/RoSD_SAR/edit/main)


## Installation  


- <span> Set up environment
 
conda create -n RoSD_SAR python=3.7  

conda activate RoSD_SAR

- <span> install pytorch
 
conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch -c conda-forge

- <span> Install
  
git clone https://github.com/douling843/RoSD_SAR.git  

cd RoSD_SAR  

pip install -r requirements/build.txt  

cd mmcv  

MMCV_WITH_OPS=1 pip install -e .  

cd ..  

pip install -e .


##  Generate noisy annotations: (e.g., 40% noise)
python ./utils/gen_noisy_ssdd.py --box_noise_level 0.4

## Train
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_ssdd_RoSD_SAR.py --work-dir='work_dir/ssdd/faster_rcnn_r50_fpn_1x_ssdd_RoSD_SAR' 

## Inference
python tools/train.py configs/faster_rcnn/faster_rcnn_r50_fpn_1x_ssdd_RoSD_SAR.py  work_dirs/ssdd/faster_rcnn_r50_fpn_1x_ssdd_RoSD_SAR/epoch_12.pth --show-dir work_dirs/vis/ssdd/RoSD_SAR


🤝  ## Acknowledgement  📫

This repository is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [OA-MIL](https://github.com/cxliu0/OA-MIL).



