# RoSD_SAR



# env
conda create -n RoSD_SAR python=3.7
conda activate RoSD_SAR

# install pytorch
conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch -c conda-forge


# clone 
git clone https://github.com/douling843/RoSD_SAR.git
cd RoSD_SAR

# install dependecies
pip install -r requirements/build.txt

# install mmcv (will take a while to process)
cd mmcv
MMCV_WITH_OPS=1 pip install -e . 

# install RoSD_SAR
cd ..
pip install -e .
