module add cuda-10.1
module add conda-modules-py37

conda activate drawnUI-conda
cd /storage/plzen1/home/vyskocj/ImageCLEFdrawnUI2021

# continue training ([--resume, -R] and [--checkpoint, -C] parameters must be added) the "faster_rcnn_X_101_32x8d_FPN_3x" model
# with batch size of 1 and accumulate gradient of 4 and learning rate of 0.0025 for 12 epochs
# After the training, show 100 predictions on the Test data
python run.py -m faster_rcnn_X_101_32x8d_FPN_3x -T -P -LP 100 -DT screenshot -b 1 -a 4 -lr 0.0025 -e 12 \
              -R -C ./__OUTPUT__/model_output/wireframe/faster_rcnn_X_101_32x8d_FPN_3x_lr_0.0025_b_1_a_4_fc_256_ff_sum_bc_0_bf_2_bcd_256_bfd_1024_d_320_e_6/model_final.pth