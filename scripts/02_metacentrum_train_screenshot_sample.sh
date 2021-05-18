module add cuda-10.1
module add conda-modules-py37

conda activate drawnUI-conda
cd /storage/plzen1/home/vyskocj/ImageCLEFdrawnUI2021

# train the "faster_rcnn_X_101_32x8d_FPN_3x" model with batch size of 1 and accumulate gradient of 4
# and learning rate of 0.0025 for 6 epochs. After the training, show 100 predictions on the Test data
python run.py -m faster_rcnn_X_101_32x8d_FPN_3x -T -P -LP 100 -DT screenshot -b 1 -a 4 -lr 0.0025 -e 6