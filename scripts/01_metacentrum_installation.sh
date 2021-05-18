# RRRRR   EEEEEE   AAAA   DDDDD         MM   MM  EEEEEE
# RR  RR  EE      AA  AA  DD  DD        MMM MMM  EE    
# RRRRR   EEEE    AAAAAA  DD  DD        MM M MM  EEEE  
# RR RR   EE      AA  AA  DD  DD        MM   MM  EE    
# RR  RR  EEEEEE  AA  AA  DDDDD         MM   MM  EEEEEE

# This script contains only the installation procedure
# -> user intervention may be required, DO NOT call this script from the bash

# ======================================================
#  QQQQ   UU  UU  EEEEEE  UU  UU  EEEEEE
# QQ  QQ  UU  UU  EE      UU  UU  EE    
# QQ  QQ  UU  UU  EEEE    UU  UU  EEEEEE
# QQ  QQ  UU  UU  EE      UU  UU  EE    
#  QQQQ    UUUU   EEEEEE   UUUU   EEEEEE
#   QQ                                  
#    QQQ                                

# Interactive queue for 1 hour with 2 CPUs, 8 GB memory and 1 GPU
# (this installation works for "adan" ifiniband)
# NOTE: for running next step scripts (training the model), you can use the no interactive queue (without -I),
#       e.g.: qsub -l select=1:ncpus=2:mem=8gb:ngpus=1:ifiniband=adan -l walltime=23:59:59 -q gpu 02_metacentrum_train_screenshot_sample.sh
qsub -I -l select=1:ncpus=2:mem=8gb:ngpus=1:ifiniband=adan -l walltime=01:00:00 -q gpu

# ======================================================
# IIIIII  NN  NN   SSSSS  TTTTTT   AAAA   LL      LL    
#   II    NNN NN  SS        TT    AA  AA  LL      LL    
#   II    NNNNNN   SSSS     TT    AAAAAA  LL      LL    
#   II    NN NNN      SS    TT    AA  AA  LL      LL    
# IIIIII  NN  NN  SSSSS     TT    AA  AA  LLLLLL  LLLLLL

module add cuda-10.1
module add conda-modules-py37

module add gcc-8.3.0
module add ninja/ninja-1.10.0-gcc-8.3.0-xraep32

# create and activate "drawnUI-conda" environment
conda create -n drawnUI-conda python=3.6
conda activate drawnUI-conda

# temp dir to not recieve error message: "Disk quota exceeded"
export TMPDIR=/storage/plzen1/home/$USER/condatmp
mkdir $TMPDIR

# go to the local directory and clone the project
cd /storage/plzen1/home/$USER/
git clone https://github.com/vyskocj/ImageCLEFdrawnUI2021.git
cd ImageCLEFdrawnUI2021

# installation of needed packages
conda install numpy                 # it is missing in conda at the default
conda install ninja                 # better to have it
pip install opencv-python           # for ImageCLEFdrawnUI2021 scripts

# installation of needed packages for detectron2
mkdir extern
cd extern
git clone https://github.com/facebookresearch/detectron2.git                                                    # clone detectron2
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html   # pytorch with cuda-10.1
pip install -e detectron2                                                                                       # detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html            # prebuild

# directory for data
cd ..
mkdir data

# NOTE: you have to add the data manually, the used structure is:
# ./data/
# ├── screenshot_development_set/
# │   ├── train/
# │   │   ├── images/
# │   │   │   └── <...>.jpg
# │   │   └── train_set.json
# │   └── validation/
# │       ├── images/
# │       │   └── <...>.jpg
# │       └── val_set.json
# ├── screenshot_test_set/
# │   └── test/images/
# │       └── <...>.jpg
# ├── wireframe_development_set/
# │   ├── images/
# │   │   └── <...>.jpg
# │   └── development_set.json
# └── wireframe_test_set/
#     └── test/
#         └── <...>.jpg

# ======================================================
# DDDDD    OOOO   NN  NN  EEEEEE
# DD  DD  OO  OO  NNN NN  EE    
# DD  DD  OO  OO  NNNNNN  EEEE  
# DD  DD  OO  OO  NN NNN  EE    
# DDDDD    OOOO   NN  NN  EEEEEE

exit
