# RoSD_SAR



conda create -n RoSD_SAR python=3.7
conda activate RoSD_SAR


conda install pytorch==1.10.0 torchvision==0.11.0 -c pytorch -c conda-forge


git clone https://github.com/douling843/RoSD_SAR.git
cd RoSD_SAR


pip install -r requirements/build.txt

cd mmcv
MMCV_WITH_OPS=1 pip install -e . 

cd ..
pip install -e .
