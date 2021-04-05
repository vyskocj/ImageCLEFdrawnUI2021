import os
import copy
import time
import argparse

import cv2
import torch
import numpy as np

from random import uniform
from shutil import copyfile
from datetime import datetime

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import hooks, DefaultPredictor
from tools.train_net import Trainer
from detectron2.engine.train_loop import SimpleTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset

from src import dataset_tool


CONFIG = {
    "wireframe": {
        "lr_decay": 0.1,             # gamma
        "lr_decay_epoch": []         # epochs (can be real number)
    },
    "screenshot": {
        "resize_ratio": 0.8,
        "resize_ratio_delta": 0.1,   # just for training
        "max_img_size": 2000,
        "lr_decay": 0.5,             # gamma
        "lr_decay_epoch": [10, ]     # epochs (can be real number)
    },
    "checkpoint_period": 5,          # epoch
    "eval_period": 1,                # epoch
    "output_path": os.path.join(
        dataset_tool.CONFIG["output_path"], "model_output"
    )
}


def custom_mapper_wf(dataset_dict):
    """
    Mapper for wireframe task

    :param dataset_dict: annotation in Detectron2 Dataset format.
    :return: dataset_dict
    """
    # Rewritten DatasetMapper (detectron2.data.dataset_mapper.py: __call__)
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format=cfg.INPUT.FORMAT)
    utils.check_image_size(dataset_dict, image)

    image, transforms = T.apply_transform_gens([
        # from detectron2.data.detection_utils.py: build_augmentation
        T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
        # lets preserve RandomFlip - CANNOT be used when horizontal and vertical are False
        # T.RandomFlip(horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal", vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
        #              prob=0.5 if cfg.INPUT.RANDOM_FLIP != "none" else 0),
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


def custom_mapper_ss_train(dataset_dict):
    """
    Mapper for screenshot task

    :param dataset_dict: annotation in Detectron2 Dataset format.
    :return: dataset_dict
    """
    # Rewritten DatasetMapper (detectron2.data.dataset_mapper.py: __call__)
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format=cfg.INPUT.FORMAT)
    utils.check_image_size(dataset_dict, image)

    x, y = float(CONFIG["screenshot"]["resize_ratio"]), float(CONFIG["screenshot"]["resize_ratio_delta"])
    resize_ratio = uniform(x - y, x + y)
    image, transforms = T.apply_transform_gens([
        T.Resize((
            int(image.shape[0] * resize_ratio),
            int(image.shape[1] * resize_ratio)
        )),
        T.RandomCrop("absolute_range", (CONFIG["screenshot"]["max_img_size"], CONFIG["screenshot"]["max_img_size"])),
        # lets preserve RandomFlip - CANNOT be used when horizontal and vertical are False
        # T.RandomFlip(horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal", vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
        #              prob=0.5 if cfg.INPUT.RANDOM_FLIP != "none" else 0),
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


def custom_mapper_ss_test(dataset_dict):
    """
    Mapper for screenshot task

    :param dataset_dict: annotation in Detectron2 Dataset format.
    :return: dataset_dict
    """
    # Rewritten DatasetMapper (detectron2.data.dataset_mapper.py: __call__)
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format=cfg.INPUT.FORMAT)
    utils.check_image_size(dataset_dict, image)

    image, transforms = T.apply_transform_gens([
        T.Resize((
            int(image.shape[0] * CONFIG["screenshot"]["resize_ratio"]),
            int(image.shape[1] * CONFIG["screenshot"]["resize_ratio"])
        ))
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


class CustomSimpleTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, accumulated_batch_size):
        super().__init__(model, data_loader, optimizer)

        self.act_step = 0
        self.accumulated_batch_size = accumulated_batch_size

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        loss = sum(loss_dict.values()) / self.accumulated_batch_size
        loss.backward()

        self._write_metrics(loss_dict, data_time)

        if (self.act_step + 1) % self.accumulated_batch_size == 0:
            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method. But it is
            suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
            """
            self.optimizer.step()

            """
            If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer.zero_grad()

            self.act_step = 0
        else:
            self.act_step += 1


class CustomTrainerWF(Trainer):
    """
    Trainer for wireframe task
    """
    def __init__(self, cfg, accumulate_batch_size):
        super().__init__(cfg)

        self._trainer = CustomSimpleTrainer(self.model, self.data_loader, self.optimizer, accumulate_batch_size)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_wf)


class CustomTrainerSS(Trainer):
    """
    Trainer for screenshot task
    """
    def __init__(self, cfg, accumulate_batch_size):
        super().__init__(cfg)

        self._trainer = CustomSimpleTrainer(self.model, self.data_loader, self.optimizer, accumulate_batch_size)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper_ss_train)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper_ss_test)


class CustomPredictorSS(DefaultPredictor):
    """
    Predictor for screenshot task
    """
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = T.Resize((
                int(height * CONFIG["screenshot"]["resize_ratio"]),
                int(width * CONFIG["screenshot"]["resize_ratio"])
            )).get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


def reg_dataset(train_imgs, train_json, valid_imgs, valid_json):
    register_coco_instances("local_dataset_train", {}, train_json, train_imgs)
    register_coco_instances("local_dataset_valid", {}, valid_json, valid_imgs)


def get_cfg_local(model_name, dataset_name, base_lr, batch_size, acum_batch_size, fpn_out_channels, fpn_fuse_type,
                  rbh_num_conv, rbh_num_fc, rbh_conv_dim, rbh_fc_dim, epochs, num_train_imgs, num_classes, maxdets,
                  checkpoint, output_dir_suffix=None):
    # initialize the configuration
    ret_cfg = get_cfg()

    # model specification
    ret_cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}.yaml"))
    ret_cfg.MODEL.WEIGHTS = checkpoint if checkpoint != "" and os.path.exists(checkpoint) \
        else model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}.yaml")
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
    ret_cfg.INPUT.RANDOM_FLIP = "none"

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
    ret_cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000 / acum_batch_size
    ret_cfg.SOLVER.WARMUP_ITERS = 1000 * acum_batch_size
    ret_cfg.SOLVER.GAMMA = CONFIG[dataset_name]["lr_decay"]       # correlation with STEPS and BASE_LR
    ret_cfg.SOLVER.STEPS = [                                      # learning rate decay in STEPS epochs by GAMMA
        int(e * num_train_imgs / batch_size) for e in CONFIG[dataset_name]["lr_decay_epoch"]
    ]
    ret_cfg.SOLVER.IMS_PER_BATCH = batch_size
    ret_cfg.SOLVER.MAX_ITER = int(epochs * num_train_imgs / batch_size)
    ret_cfg.SOLVER.CHECKPOINT_PERIOD = int(CONFIG["checkpoint_period"] * num_train_imgs / batch_size)  # save checkpoint

    # output path
    suffix = "" if (output_dir_suffix is None or output_dir_suffix == "") else f"_{output_dir_suffix}"
    ret_cfg.OUTPUT_DIR = os.path.join(CONFIG["output_path"], dataset_name, f"{model_name}{suffix}")
    # check if output path exists
    output_orig = ret_cfg.OUTPUT_DIR
    idx = 1
    while os.path.exists(ret_cfg.OUTPUT_DIR):
        ret_cfg.OUTPUT_DIR = f"{output_orig}({idx})"
        idx += 1

    return ret_cfg


def get_cfg_after_train(num_imgs, batch_size):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")           # load the weights for Predictor
    cfg.DATASETS.TRAIN += cfg.DATASETS.TEST                                       # add validation dataset

    num_lr_decays = max(
        n if step <= cfg.SOLVER.MAX_ITER else 0
        for n, step in enumerate(cfg.SOLVER.STEPS, start=1)
    )
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * (cfg.SOLVER.GAMMA ** num_lr_decays)  # continue with the learning rate
    cfg.SOLVER.WARMUP_ITERS = 0                                                    # remove warmup
    cfg.SOLVER.MAX_ITER = int(2 * num_imgs / batch_size)                           # train for 2 epochs
    cfg.OUTPUT_DIR += "_afterTrain"                                                # define a new output directory

    # check if output path exists
    output_orig = cfg.OUTPUT_DIR
    idx = 1
    while os.path.exists(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = f"{output_orig}({idx})"
        idx += 1


def show_predictions(predictor, imgs_dir, output_dir, category_names, lim_predictions):
    # Store the category names
    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = category_names

    print("Saving model predictions...")
    time = datetime(2020, 1, 1)
    num_imgs = len(os.listdir(imgs_dir))
    for img_id, img_name in enumerate(os.listdir(imgs_dir), start=1):
        # read an image and predict
        img = cv2.imread(os.path.join(imgs_dir, img_name))
        outputs = predictor(img)

        # visualize the predictions
        v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # save an image
        cv2.imwrite(os.path.join(output_dir, img_name), out.get_image()[:, :, ::-1])

        # print status of processing
        time_now = datetime.now()
        if (time_now - time) > dataset_tool.CONFIG["print_status"]:
            print(f"{img_id}/{num_imgs}")
            time = time_now

        if lim_predictions == img_id:
            # img_id starts from 1 -> when lim_predictions == 0, no limitation is used
            break

    print(f"Predictions were successfully saved: {output_dir}")


if __name__ == "__main__":
    # Parse commandline
    parser = argparse.ArgumentParser(description='Dataset tools for competition "ImageCLEF DrawnUI 2021".')

    # Optional arguments
    parser.add_argument(
        '-m', '--model', default="faster_rcnn_R_50_FPN_3x",
        help='Model to be trained. See names in "/detectron2/model_zoo/configs/COCO-Detection" directory.'
    )
    parser.add_argument('-lr', '--base_lr', default=.0025, help='Learning rate.')
    parser.add_argument('-b', '--batch_size', default=4, help='Batch size.')
    parser.add_argument('-a', '--accum_batch_size', default=1, help='Batch size of accumulate gradient.')

    parser.add_argument('-fc', '--fpn_channels', default=256, help='Number of Feature Pyramid Network channels.')
    parser.add_argument('-ff', '--fpn_fuse_type', default="sum", help='A fuse type of Feature Pyramid Network:'
                                                                      ' ["sum", "avg"].')

    parser.add_argument('-bc', '--rbh_num_conv', default=0, help='Number of convolution layers in the Roi Box Head.')
    parser.add_argument('-bf', '--rbh_num_fc', default=2, help='Number of fully-connected layers in the Roi Box Head.')
    parser.add_argument('-bcd', '--rbh_conv_dim', default=256, help='Convolution layers dimension in the RBH.')
    parser.add_argument('-bfd', '--rbh_fc_dim', default=1024, help='Fully-connected layers dimension in the RBH.')

    parser.add_argument('-d', '--maxdets', default=320, help='Maximum number of detections.')
    parser.add_argument('-e', '--epochs', default=200, help='Number of training epochs.')

    # Train / test arguments
    parser.add_argument('-T', '--train', action='store_true', help='Train the network.')
    parser.add_argument('-C', '--checkpoint', default='', help='Path to the checkpoint of your model'
                                                               ' (model_final.pth).')
    parser.add_argument('-R', '--resume', action='store_true', help='Resume training from last checkpoint.'
                                                                    ' Parameter -C is needed!')
    parser.add_argument('-P', '--predict', action='store_true', help='Show predictions of the model. If parameter -T is'
                                                                     ' used, predictions fot the Test set are stored in'
                                                                     ' the same directory as the trained model.')
    parser.add_argument('-DT', '--dataset', default='wireframe',
                        help='Dataset for training: ["wireframe", "screenshot"].')
    parser.add_argument('-DP', '--data_predict', default='test',
                        help='The set which predictions shall be visualized. Relevant only for parameter -P (parameter'
                             ' -T must not be used), supported values: ["train", "valid", "test"].')  # TODO: dodělat
    parser.add_argument('-LP', '--lim_predict', default=0, help='Limit the number of visualized predictions. Relevant'
                                                                ' with parameter -P (parameter -T can also be used).'
                                                                ' If "0" is passed, predictions for all images are'
                                                                ' visualized.')

    # Parsing arguments
    args = parser.parse_args()

    # =================================================================================================================

    # get the dataset information
    path_to_imgs, path_to_json = list(), list()
    dataset_suff = list()
    split_data = ""
    eval_json, eval_imgs = "", ""
    if args.dataset == "wireframe":
        path_to_imgs = [dataset_tool.CONFIG["wireframe_data"]]
        path_to_json = [dataset_tool.CONFIG["wireframe_json"]]
        dataset_suff = [""]
        split_data = "true"

        eval_imgs = dataset_tool.CONFIG["wireframe_eval_data"]
    elif args.dataset == "screenshot":
        path_to_imgs = [dataset_tool.CONFIG["screenshot_train_data"], dataset_tool.CONFIG["screenshot_valid_data"]]
        path_to_json = [dataset_tool.CONFIG["screenshot_train_json"], dataset_tool.CONFIG["screenshot_valid_json"]]
        dataset_suff = ["_train", "_valid"]
        split_data = "false"

        eval_imgs = dataset_tool.CONFIG["screenshot_eval_data"]
    else:
        Exception("You set wrong --dataset argument! Expected is one of ['wireframe', 'screenshot'],"
                  f" actual is {args.dataset}")

    # generate coco annotation files
    coco = dict()
    num_train_imgs = 0
    num_valid_imgs = 0
    num_catetogies = 0
    for i in range(len(path_to_imgs)):
        coco = dataset_tool.make_coco(path_to_json[i], f"{args.dataset}{dataset_suff[i]}", path_to_imgs[i], split_data)
        if i == 0:
            num_train_imgs = len(coco[0]["images"])
            num_catetogies = len(coco[0]["categories"])
        num_valid_imgs = len(coco[-1]["images"])

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

    # register the datasets
    reg_dataset(path_to_train_imgs, path_to_train_json, path_to_valid_imgs, path_to_valid_json)
    if os.path.exists(eval_imgs):
        coco_eval, eval_json = dataset_tool.make_coco_eval(f"{args.dataset}_eval", eval_imgs, coco[0]["categories"])
        register_coco_instances("local_dataset_eval", {}, eval_json, eval_imgs)
    else:
        coco_eval = dict()  # dummy

    # get configuration and prepare an output directory
    out_dir_suff = f"lr_{args.base_lr}_b_{args.batch_size}_a_{args.accum_batch_size}_fc_{args.fpn_channels}" \
                   f"_ff_{args.fpn_fuse_type}_bc_{args.rbh_num_conv}_bf_{args.rbh_num_fc}_bcd_{args.rbh_conv_dim}" \
                   f"_bfd_{args.rbh_fc_dim}_d_{args.maxdets}_e_{args.epochs}"
    cfg = get_cfg_local(
        args.model, args.dataset, float(args.base_lr), int(args.batch_size), int(args.accum_batch_size),
        int(args.fpn_channels), args.fpn_fuse_type, int(args.rbh_num_conv), int(args.rbh_num_fc),
        int(args.rbh_conv_dim), int(args.rbh_fc_dim), int(args.epochs), num_train_imgs=num_train_imgs,
        num_classes=num_catetogies, maxdets=int(args.maxdets), checkpoint=args.checkpoint,
        output_dir_suffix=out_dir_suff
    )
    os.makedirs(cfg.OUTPUT_DIR)
    if args.resume is True:
        # copy last_checkpoint to continue training
        copyfile(
            os.path.join(os.path.dirname(args.checkpoint), "last_checkpoint"),
            os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
        )
        copyfile(
            os.path.join(os.path.dirname(args.checkpoint), "model_final.pth"),
            os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        )

    # train the model
    if args.train is True:
        for phase in ["training", "overfitting"]:
            if phase == "overfitting":
                # load new config including validation dataset for training
                get_cfg_after_train(num_train_imgs + num_valid_imgs, int(args.batch_size))
                os.makedirs(cfg.OUTPUT_DIR)

            # get trainer
            if args.dataset == "wireframe":
                trainer = CustomTrainerWF(cfg, int(args.accum_batch_size))
            else:  # args.dataset == "screenshot"
                trainer = CustomTrainerSS(cfg, int(args.accum_batch_size))

            # add TTA evaluation after training
            if args.dataset != "screenshot":
                # TTA is not supported for screenshot set yet
                trainer.register_hooks(
                    [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
                )

            # train the model from zero or continue training
            trainer.resume_or_load(resume=False if ((phase == "overfitting") or (args.resume is False)) else True)
            trainer.train()

            # ==========================================================================================================
            if os.path.exists(eval_imgs):
                # generate submission file
                eval_output = os.path.join(cfg.OUTPUT_DIR, "submission")
                os.makedirs(eval_output, exist_ok=True)

                # create COCO evaluator
                evaluator = COCOEvaluator("local_dataset_eval", ("bbox",), False, output_dir=eval_output)
                eval_loader = trainer.build_test_loader(cfg, "local_dataset_eval")
                inference_on_dataset(trainer.model, eval_loader, evaluator)  # it returns AP measurements

                # make submission file
                dataset_tool.make_submission_file(
                    os.path.join(eval_output, "coco_instances_results.json"),
                    coco_eval,
                    eval_output
                )

                # create a new directory for predictions
                eval_output = os.path.join(eval_output, "predictions")
                os.makedirs(eval_output, exist_ok=True)

                # visualise predictions
                if args.predict is True:
                    categories = [c["name"] for c in coco_eval["categories"]]
                    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")   # weights from the checkpoint
                    if args.dataset == "wireframe":
                        show_predictions(DefaultPredictor(cfg), eval_imgs, eval_output, categories,
                                         int(args.lim_predict))
                    else:  # args.dataset == "screenshot"
                        show_predictions(CustomPredictorSS(cfg), eval_imgs, eval_output, categories,
                                         int(args.lim_predict))
