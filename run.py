import os
import copy
import argparse

import cv2
import torch
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.engine import hooks, DefaultPredictor
from tools.train_net import Trainer

from src import dataset_tool


CONFIG = {
    "checkpoint_period": 50,  # epoch
    "eval_period": 1,         # epoch
    "output_path": os.path.join(
        dataset_tool.CONFIG["output_path"], "model_output"
    )
}


def custom_mapper(dataset_dict):
    # Rewritten DatasetMapper (detectron2.data.dataset_mapper.py: __call__)
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    utils.check_image_size(dataset_dict, image)

    image, transforms = T.apply_transform_gens([
        # from detectron2.data.detection_utils.py: build_augmentation
        T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
        # lets preserve RandomFlip
        T.RandomFlip(horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal", vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                     prob=0.5),
        # new augmentations
        T.RandomApply(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
                      prob=0.5),
        T.RandomApply(T.RandomContrast(intensity_min=0.5, intensity_max=1.5),
                      prob=0.5),
        T.RandomApply(T.RandomSaturation(intensity_min=0.5, intensity_max=1.5),
                      prob=0.5)
    ], image)

    image_shape = image.shape[:2]  # h, w
    # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
    # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
    # Therefore it's important to use torch.Tensor.
    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image_shape)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image_shape)

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class CustomTrainer(Trainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


def reg_dataset(train_imgs, train_json, valid_imgs, valid_json):
    register_coco_instances("local_dataset_train", {}, train_json, train_imgs)
    register_coco_instances("local_dataset_valid", {}, valid_json, valid_imgs)


def get_cfg_local(model_name, base_lr, batch_size, fpn_out_channels, fpn_fuse_type, rbh_num_conv, rbh_num_fc,
                  rbh_conv_dim, rbh_fc_dim, epochs, num_train_imgs, num_classes, maxdets, output_dir_suffix=None):
    # initialize the configuration
    ret_cfg = get_cfg()

    # model specification
    ret_cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}.yaml"))
    ret_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}.yaml")
    ret_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # FPN
    ret_cfg.MODEL.FPN.OUT_CHANNELS = fpn_out_channels
    ret_cfg.MODEL.FPN.FUSE_TYPE = fpn_fuse_type

    # Box Head
    ret_cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = rbh_num_conv  # number of convolutional layers behind RoI pooling
    ret_cfg.MODEL.ROI_BOX_HEAD.NUM_FC = rbh_num_fc      # number of fully connected layers behind conv layers
    ret_cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = rbh_conv_dim  # dimension of all conv layers in a box head
    ret_cfg.MODEL.ROI_BOX_HEAD.FC_DIM = rbh_fc_dim      # dimension of all fc layers in a box head and box predictor

    # input
    ret_cfg.INPUT.RANDOM_FLIP = "horizontal"

    # dataset settings
    ret_cfg.DATASETS.TRAIN = ("local_dataset_train",)
    ret_cfg.DATASETS.TEST = ("local_dataset_valid",)
    ret_cfg.DATALOADER.NUM_WORKERS = 2

    # evaluate model at the end of each epoch
    ret_cfg.TEST.EVAL_PERIOD = int(CONFIG["eval_period"] * num_train_imgs / batch_size)
    ret_cfg.VIS_PERIOD = ret_cfg.TEST.EVAL_PERIOD
    ret_cfg.TEST.DETECTIONS_PER_IMAGE = maxdets
    # TODO: vypnuté pro trénování
    # ret_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # to eliminate predictions with probability under 70 %

    # learning settings
    ret_cfg.SOLVER.BASE_LR = base_lr
    ret_cfg.SOLVER.GAMMA = 0.1       # correlation with STEPS and BASE_LR
    ret_cfg.SOLVER.STEPS = [         # learning rate decay in [100, 150, 175] epochs by GAMMA
        int(100 * num_train_imgs / batch_size), int(150 * num_train_imgs / batch_size),
        int(175 * num_train_imgs / batch_size)
    ]
    ret_cfg.SOLVER.IMS_PER_BATCH = batch_size
    ret_cfg.SOLVER.MAX_ITER = int(epochs * num_train_imgs / batch_size)
    ret_cfg.SOLVER.CHECKPOINT_PERIOD = int(CONFIG["checkpoint_period"] * num_train_imgs / batch_size)  # save checkpoint

    # output path
    suffix = "" if (output_dir_suffix is None or output_dir_suffix == "") else f"_{output_dir_suffix}"
    ret_cfg.OUTPUT_DIR = os.path.join(CONFIG["output_path"], f"{model_name}{suffix}")

    return ret_cfg


def predict(predictor, imgs_dir, cfg, visualize=True):
    max_size = 1333
    size = 800

    for img_name in os.listdir(imgs_dir):
        img = cv2.imread(os.path.join(imgs_dir, img_name))

        outputs = predictor(img)
        if visualize is True:
            # TODO: https://github.com/facebookresearch/detectron2/issues/326
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow(img_name, out.get_image()[:, :, ::-1])

        cv2.waitKey(0)


if __name__ == "__main__":
    # Parse commandline
    parser = argparse.ArgumentParser(description='Dataset tools for competition "ImageCLEF DrawnUI 2021".')

    # Optional arguments
    parser.add_argument(
        '-m', '--model', default="faster_rcnn_R_50_FPN_3x",
        help='Model to be trained. See names in "/detectron2/model_zoo/configs/COCO-Detection" directory.'
    )
    parser.add_argument('-lr', '--base_lr', default=.00025, help='Learning rate.')
    parser.add_argument('-b', '--batch_size', default=4, help='Batch size.')
    parser.add_argument('-fc', '--fpn_channels', default=256, help='Number of proposals for detection.')
    parser.add_argument('-ff', '--fpn_fuse_type', default="sum", help='Number of proposals for detection.')
    parser.add_argument('-bc', '--rbh_num_conv', default=0, help='Number of proposals for detection.')
    parser.add_argument('-bf', '--rbh_num_fc', default=2, help='Number of proposals for detection.')
    parser.add_argument('-bcd', '--rbh_conv_dim', default=256, help='Number of proposals for detection.')
    parser.add_argument('-bfd', '--rbh_fc_dim', default=1024, help='Number of proposals for detection.')
    parser.add_argument('-d', '--maxdets', default=320, help='Maximum number of detections.')
    parser.add_argument('-e', '--epochs', default=200, help='Number of training epochs.')

    # Train / test arguments
    parser.add_argument('-T', '--train', action='store_true', help='Learning rate.')
    parser.add_argument('-P', '--predict', action='store_true', help='Learning rate.')
    parser.add_argument('-C', '--checkpoint', default='', help='Path to a checkpoint of your model.')
    parser.add_argument('-DT', '--dataset', default='wireframe',
                        help='Dataset for training: ["wireframe", "screenshot"].')
    parser.add_argument('-DP', '--data_predict', default='',
                        help='Path to the data for predictions. If not set, validation set is used.')

    # Parsing arguments
    args = parser.parse_args()

    # =================================================================================================================
    # check arguments
    if args.predict is True and args.train is False and args.checkpoint == "":
        Exception("There is no way to predict when --train is False and --checkpoint is not set!")

    # get the dataset information
    path_to_imgs, path_to_json = list(), list()
    split_data = ""
    if args.dataset == "wireframe":
        path_to_imgs = [dataset_tool.CONFIG["wireframe_data"]]
        path_to_json = [dataset_tool.CONFIG["wireframe_json"]]
        split_data = "true"
    elif args.dataset == "screenshot":
        path_to_imgs = [dataset_tool.CONFIG["screenshot_train_data"], dataset_tool.CONFIG["screenshot_valid_data"]]
        path_to_json = [dataset_tool.CONFIG["screenshot_train_json"], dataset_tool.CONFIG["screenshot_valid_json"]]
        split_data = "false"
    else:
        Exception("You set wrong --dataset argument! Expected is one of ['wireframe', 'screenshot'],"
                  f" actual is {args.dataset}")

    # generate coco annotation files
    coco = dict()
    for i in range(len(path_to_imgs)):
        coco = dataset_tool.make_coco(path_to_json[i], args.dataset, path_to_imgs[i], split_data)

    # copy data into new directory if data was split before (i.e. wireframe dataset was set to training)
    # and define paths (to images and json files)
    if len(coco) == 2:
        path_to_train_imgs = os.path.join(dataset_tool.CONFIG["output_path"], "copy_data", f"{args.dataset}_train")
        path_to_valid_imgs = os.path.join(dataset_tool.CONFIG["output_path"], "copy_data", f"{args.dataset}_valid")
        if not os.path.exists(path_to_train_imgs):
            for i, dataset_type in enumerate(["train", "valid"]):
                dataset_tool.copy_data(coco[i], path_to_imgs[0], f"{args.dataset}_{dataset_type}")
    else:
        path_to_train_imgs = path_to_imgs[0]
        path_to_valid_imgs = path_to_imgs[1]

    path_to_train_json = os.path.join(dataset_tool.CONFIG["output_path"], "make_coco", f"coco_{args.dataset}_train.json")
    path_to_valid_json = os.path.join(dataset_tool.CONFIG["output_path"], "make_coco", f"coco_{args.dataset}_valid.json")

    # register the dataset
    reg_dataset(path_to_train_imgs, path_to_train_json, path_to_valid_imgs, path_to_valid_json)

    # get configuration and make an output directory
    out_dir_suff = f"lr_{args.base_lr}_b_{args.batch_size}_fc_{args.fpn_channels}_ff_{args.fpn_fuse_type}" \
                   f"_bc_{args.rbh_num_conv}_bf_{args.rbh_num_fc}_bcd_{args.rbh_conv_dim}_bfd_{args.rbh_fc_dim}" \
                   f"_d_{args.maxdets}_e_{args.epochs}"
    cfg = get_cfg_local(
        args.model, float(args.base_lr), int(args.batch_size), int(args.fpn_channels), args.fpn_fuse_type,
        int(args.rbh_num_conv), int(args.rbh_num_fc), int(args.rbh_conv_dim), int(args.rbh_fc_dim), int(args.epochs),
        num_train_imgs=len(coco[0]["images"]), num_classes=len(coco[0]["categories"]), maxdets=int(args.maxdets),
        output_dir_suffix=out_dir_suff
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # train the model
    if args.train is True:
        trainer = CustomTrainer(cfg)
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
        trainer.resume_or_load(resume=False)
        trainer.train()

    # get predictions
    if args.predict is True:
        if args.train is True:
            model = trainer.model
        else:
            model = DefaultPredictor(cfg)
            DetectionCheckpointer(model.model).load(args.checkpoint)

        if args.data_predict == "":
            data_path = path_to_valid_imgs
        else:
            data_path = args.data_predict

        predict(model, data_path, cfg)
