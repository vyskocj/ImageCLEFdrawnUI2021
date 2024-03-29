from PIL import Image, ImageFilter, ImageStat
from datetime import datetime, timedelta
from shutil import copyfile

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import json
import os

import argparse


CONFIG = {
    "screenshot_valid_data": os.path.join(
        os.path.dirname(__file__), "..", "data", "screenshot_development_set", "validation", "images"
    ),
    "screenshot_valid_json": os.path.join(
        os.path.dirname(__file__), "..", "data", "screenshot_development_set", "validation", "val_set.json"
    ),
    "screenshot_train_data": os.path.join(
        os.path.dirname(__file__), "..", "data", "screenshot_development_set", "train", "images"
    ),
    "screenshot_train_json": os.path.join(
        os.path.dirname(__file__), "..", "data", "screenshot_development_set", "train", "train_set.json"
    ),
    "screenshot_eval_data": os.path.join(
        os.path.dirname(__file__), "..", "data", "screenshot_test_set", "test", "images"
    ),
    "wireframe_data": os.path.join(
        os.path.dirname(__file__), "..", "data", "wireframe_development_set", "images"
    ),
    "wireframe_json": os.path.join(
        os.path.dirname(__file__), "..", "data", "wireframe_development_set", "development_set.json"
    ),
    "wireframe_eval_data": os.path.join(
        os.path.dirname(__file__), "..", "data", "wireframe_test_set", "test"
    ),
    "output_path": os.path.join(
        os.path.dirname(__file__), "..", "__OUTPUT__"
    ),
    "print_status": timedelta(seconds=5),
    "train_ratio": .85,                   # used to divide data into a training and validation set
    "plt_keys": [["train_loss", "test_loss"], ["train_loss_ce", "test_loss_ce"], ["train_loss_bbox", "test_loss_bbox"]]
}


def decode(img_metadata):
    return img_metadata["file"], img_metadata["width"], img_metadata["height"], img_metadata["annotations"]


def decode_bbox(box, img_width, img_height):
    # coco format defines [x, y, width, height]
    return box[1] * img_width, box[0] * img_height, box[3] * img_width, box[2] * img_height


def encode_bbox(box, img_width, img_height):
    # drawnUI challenge format defines [y, x, height, width] as relative
    return box[1] / img_height, box[0] / img_width, box[3] / img_height, box[2] / img_width


def vis_data(json_path, data_path, show=False, save=True, set_name="data"):
    if show is False and save is False:
        print("WARNING: Function 'vis_data' is dummy! Please set argument 'show' or 'save' to True...")

    with open(json_path, "r") as json_file:
        # load json file and randomize
        metadata_list = json.load(json_file)
        random.shuffle(metadata_list)

        # additional info - used if argument 'show' is False
        num_imgs = len(metadata_list)
        cur_img = 0
        time = datetime(2020, 1, 1)

        # define output if argument 'save' is True
        output_path = os.path.join(CONFIG["output_path"], "vis_data", set_name)
        if save and not os.path.exists(output_path):
            os.makedirs(output_path)

        # info how to exit program
        if show:
            print("Press enter to continue, type string to exit...")

        # define dpi
        dpi = 80
        for metadata in metadata_list:
            file, _, _, annot = decode(metadata)

            with Image.open(os.path.join(data_path, file)) as img:
                # plot image
                fig = plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
                plt.imshow(img)
                ax = plt.gca()  # get current reference

                legend_lbls = list()
                color_map = plt.cm.get_cmap('hsv', len(annot) + 1)
                for i in range(0, len(annot)):
                    # get bbox coordinates
                    left, top, width, height = decode_bbox(annot[i]["box"], img.size[0], img.size[1])

                    # draw bbox
                    rect = patches.Rectangle((left, top), width, height, linewidth=1, edgecolor=color_map(i),
                                             facecolor='none')
                    ax.add_patch(rect)

                    # append label name
                    legend_lbls.append(annot[i]["detectionString"])

                # plt.legend(legend_lbls)
                plt.axis('off')

                # show image and then wait, or print status of img processing
                if show:
                    print(f"img: {file}, w: {img.size[0]}, h: {img.size[1]}")
                    plt.show()
                else:
                    time_now = datetime.now()
                    if (time_now - time) > CONFIG["print_status"]:
                        print(f"{cur_img}/{num_imgs}")
                        time = time_now
                    cur_img += 1

                # save img
                if save:
                    fig.savefig(os.path.join(output_path, file), dpi=dpi)

                # close plot / save memory
                plt.close()

            # exit program
            if show:
                btn = input()
                if btn != "":
                    exit(0)

        # print exit status
        if save:
            print(f"You can find files in: {output_path}")


def get_moving_threshold(img_size):
    # should be used only for screenshot task
    # The thresholds are used for IMAGES!
    if img_size <= 500**2:
        return 2.5
    if img_size <= 600**2:           # threshold max 3.5
        return 3.5
    elif img_size <= 900**2:         # threshold max 2.8
        return 2.8
    elif img_size <= 1200**2:        # threshold max 2.5
        return 2.5
    else:                            # threshold max 0.8
        return 0.8


def is_homo(image, threshold, is_bbox, image_name=""):
    # convert the image to greyscale
    img = image.convert("L")

    # detect edges in the image
    img = img.filter(ImageFilter.FIND_EDGES)

    if is_bbox is False:
        # get threshold for image
        threshold = get_moving_threshold(image.size[0] * image.size[1])

    # compute mean of edges and return if the image is homogeneous
    img_mean = ImageStat.Stat(img).mean[0]
    is_homogeneous = img_mean < threshold

    # save image
    if image_name != "" and is_homogeneous:
        img_size = image.size[0] * image.size[1]
        if img_size <= 25**2:
            img_size = "max_25x25"
        elif img_size <= 50**2:
            img_size = "max_50x50"
        elif img_size <= 100**2:
            img_size = "max_100x100"
        elif img_size <= 200**2:
            img_size = "max_200x200"
        elif img_size <= 300**2:
            img_size = "max_300x300"
        elif img_size <= 400**2:
            img_size = "max_400x400"
        elif img_size <= 500**2:
            img_size = "max_500x500"
        elif img_size <= 600**2:
            img_size = "max_600x600"
        elif img_size <= 900**2:
            img_size = "max_900x900"
        elif img_size <= 1000**2:
            img_size = "max_1000x1000"
        else:
            img_size = "other_size"

        output_path = os.path.join(CONFIG["output_path"], "is_homo", "bbox" if is_bbox else "img", img_size)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        image.save(os.path.join(output_path, f"{img_mean}_{image_name}_orig.png"))  # f"{image_name}_orig.png"))
        img.save(os.path.join(output_path, f"{img_mean}_{image_name}_edge.png"))  # f"{image_name}_edge.png"))

    return is_homogeneous


def make_coco(json_path, set_name=None, data_path=None, split="false", reproducibility_set=False, discard=[],
              threshold_box=-1, report_wrong_img_size=False, save_discarded=False):
    """
    Make COCO-style annotations.

    :param json_path: path to the drawnUI annotations (.json file)
    :param set_name: dataset name (optional)
    :param data_path: path to the image data (optional)
    :param split: split dataset, one of ["true", "false", "random"]. Default is "false" - data are preserved. If "true"
                  then data are split according to the "<set_name>_indexes.json" file. If "random" then data are
                  randomly split.
    :param reproducibility_set: save indexes of the split data to the "<set_name>_indexes.json" file
    :param discard: discard some annotations or whole images, supported list values: ["outlier_boxes",
                       "hmg_boxes", "hmg_imgs"]. Outlier boxes are annotations which coordinates (including
                       width/height) belongs outside the image. Homogeneous box/image is an element of similar
                       intensity (see parameter threshold_box).
    :param threshold_box: the tolerated threshold that the object is homogeneous. Edge filtering -> sum and norm
                          filtered image -> application the threshold.
    :param report_wrong_img_size: print image names with wrong sizes
    :param save_discarded: save discarded images
    :return: list of COCO annotations and output path where the annotations were saved
    """
    if len(discard) != 0:
        print(f"Discarding data for dataset: {set_name}")

    def _process(metadata_list, set_type=None):
        # initialization of coco dictionary
        coco = {
           "info": {
               "description": "COCO drawnUI2021 dataset",
                "version": "1.0",
                "year": 2021,
                "date_created": "2021/02/12"
           },
           "licenses": [],
           "images": [],
           "annotations": [],
           "categories": []
        }

        # information about category can be found also in the provided json
        categories = dict()  # key = label (class name); value = number (class index from 1 to N)

        # process the metadata
        ann_id = 1
        miss_img = 0  # missing img in the data directory
        img = None
        discarded_imgs = 0
        discarded_outlier_boxes = 0
        discarded_homogen_boxes = 0
        for img_id, metadata in enumerate(metadata_list, start=1):
            file, width, height, annot = decode(metadata)

            # check if img exists
            if data_path is not None and not os.path.exists(os.path.join(data_path, file)):
                print(f"Info: file '{file}' is missing in data directory for the '{dataset_name}' dataset...")
                miss_img += 1
                continue

            # check the image size
            if data_path is not None:
                img = Image.open(os.path.join(data_path, file))
                if width != img.size[0] or height != img.size[1]:
                    if report_wrong_img_size:
                        print(f"Image {file} have (w, h) size {img.size} since annotation refers to"
                              f" {(width, height)}")
                    width, height = img.size

            if ("hmg_imgs" in discard) and is_homo(img, -1, is_bbox=False,
                                                    image_name=file.split(".")[0] if save_discarded is True else ""):
                discarded_imgs += 1
                continue

            # append info about image
            coco["images"].append(
                {
                    "id": img_id - miss_img,  # to maintain ids from 1 to real N
                    "file_name": file,
                    "width": width,
                    "height": height
                }
            )

            ann_id_old = ann_id
            for a, ann in enumerate(annot, start=1):
                if ann["detectionString"] not in categories.keys():
                    # store label to the category
                    categories[ann["detectionString"]] = ann["detectionClass"]

                # append annotation
                x, y, w, h = decode_bbox(ann["box"], width, height)
                if ("outlier_boxes" in discard) and (((x + w - .001) > width) or ((y + h - .001) > height)):
                    # NOTE: a small numerical error may occur after calling decode_bbox (thanks to multiplication of
                    #       relative numbers) -> delta should be used to suppress false outliers
                    discarded_outlier_boxes += 1
                    continue

                if ("hmg_boxes" in discard) and is_homo(img.crop((x, y, x + w, y + h)), threshold_box, True,
                                                         f"{file.split('.')[0]}_bbox_{a}"if save_discarded else ""):
                    discarded_homogen_boxes += 1
                    continue

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": ann["detectionClass"],
                        "bbox": [x, y, w, h],
                        "area": w * h,           # dummy - DETR requires it
                        "iscrowd": 0             # needed for validation set by DETR
                        # "score": ann["score"]  # score is always 1 -> irrelevant information
                    }
                )
                ann_id += 1

            if ann_id_old == ann_id:
                coco["images"].pop(-1)

        # sort categories by id and store them in the coco dictionary
        cat_sorted = sorted(zip(
            list(categories.values()), list(categories.keys())
        ))
        for cat_id, cat_name in cat_sorted:
            coco["categories"].append(
                {
                    "id": cat_id,
                    "name": cat_name
                }
            )

        # save data, if required
        if set_name is not None:
            output_path = os.path.join(CONFIG["output_path"], "make_coco")
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if set_type is None:
                output_path = os.path.join(output_path, f"coco_{set_name}.json")
            else:
                output_path = os.path.join(output_path, f"coco_{set_name}_{set_type}.json")
            with open(output_path, "w") as coco_file:
                json.dump(coco, coco_file)
                print(f"COCO annotation file saved: {output_path}")
                print(f"Number of missing images: {miss_img}")
                if len(discard) != 0:
                    print(f"Used threshold for bbox: {threshold_box}")
                if discarded_imgs != 0:
                    print(f"Discarded images: {discarded_imgs}")
                if discarded_outlier_boxes != 0:
                    print(f"Discarded bounding box outliers: {discarded_outlier_boxes}")
                if discarded_homogen_boxes != 0:
                    print(f"Discarded homogeneous boxes: {discarded_homogen_boxes}")
                print("")

        return coco

    with open(json_path, "r") as json_file:
        # load json file
        metadata_all = json.load(json_file)

        coco_list = list()
        if str(split).lower() == "false":
            coco_list.append(_process(metadata_all))
        elif str(split).lower() == "random":
            num_train = 0
            while True:
                # shuffle indexes of data
                metadata_length = len(metadata_all)
                metadata_indexes = list(range(metadata_length))
                random.shuffle(metadata_indexes)

                num_train = int(metadata_length * CONFIG["train_ratio"])

                # split metadata to training and validation subsets
                train_set = [metadata_all[idx] for idx in metadata_indexes[:num_train]]
                valid_set = [metadata_all[idx] for idx in metadata_indexes[num_train:]]

                # create coco datasets
                coco_list.append(_process(train_set, "train"))
                coco_list.append(_process(valid_set, "valid"))

                # check if all categories in both sets are represented
                if coco_list[0]["categories"] == coco_list[1]["categories"]:
                    break
                else:
                    print("In training and validation set are not all categories represented... Generating a new set!")
                    continue

            # store indexes for allowing reproducibility
            if reproducibility_set:
                with open(os.path.join(os.path.dirname(__file__), f"{set_name}_indexes.json"), "w") as output_file:
                    metadata_dict = {
                        "train": metadata_indexes[:num_train],
                        "valid": metadata_indexes[num_train:]
                    }
                    json.dump(metadata_dict, output_file)
                    print(f"Reproducibility set was saved... {os.path.join(os.path.dirname(__file__), f'{set_name}_indexes.json')}")
        elif str(split).lower() == "true":
            with open(os.path.join(os.path.dirname(__file__), f"{set_name}_indexes.json"), "r") as idx_file:
                set_indexes = json.load(idx_file)

                # split metadata to training and validation subsets
                train_set = [metadata_all[idx] for idx in set_indexes["train"]]
                valid_set = [metadata_all[idx] for idx in set_indexes["valid"]]

                coco_list.append(_process(train_set, "train"))
                coco_list.append(_process(valid_set, "valid"))

        else:
            raise Exception("Error: Wrong 'split' argument is passed... Expected one of ['false', 'true', "
                            f"'reproducibility'], actual is '{split}'")

    return coco_list


def make_coco_eval(set_name, data_path, coco_categories):
    # initialization of coco dictionary
    coco = {
       "info": {
           "description": "COCO drawnUI2021 dataset",
            "version": "1.0",
            "year": 2021,
            "date_created": "2021/02/12"
       },
       "licenses": [],
       "images": [],
       "annotations": [],
       "categories": []
    }

    # list images
    for img_id, filename in enumerate(os.listdir(data_path), start=1):
        # check the image size
        with Image.open(os.path.join(data_path, filename)) as img:
            width, height = img.size

        # append image info
        coco["images"].append(
            {
                "id": img_id,
                "file_name": filename,
                "width": width,
                "height": height
            }
        )

    # sort categories by id and store them in the coco dictionary
    coco["categories"] = coco_categories

    # save data, if required
    output_path = ""
    if set_name is not None:
        output_path = os.path.join(CONFIG["output_path"], "make_coco")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = os.path.join(output_path, f"coco_{set_name}.json")
        with open(output_path, "w") as coco_file:
            json.dump(coco, coco_file)
            print(f"COCO evaluation file saved: {output_path}")
            print("")

    return coco, output_path


def copy_data(coco_data, data_path, set_name):
    output_path = os.path.join(CONFIG["output_path"], "copy_data", set_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f"Copying {set_name} data...")
    time = datetime(2020, 1, 1)
    num_imgs = len(coco_data["images"])
    for img_id, img in enumerate(coco_data["images"], 1):
        copyfile(
            os.path.join(data_path, img["file_name"]),
            os.path.join(output_path, img["file_name"])
        )

        # print status of processing
        time_now = datetime.now()
        if (time_now - time) > CONFIG["print_status"]:
            print(f"{img_id}/{num_imgs}")
            time = time_now

    print(f"Data was successfully copied: {output_path}")
    print("")


def get_stats(coco_data, set_name=None, relative=True):
    num_cat = len(coco_data["categories"])
    num_ann = len(coco_data["annotations"])

    # initialization of statistics dictionary
    stats = {
        "image": {
            "min": {
                "width": 65535,
                "height": 65535
            },
            "max": {
                "width": 0,
                "height": 0
            },
            "avg": {
                "width": 0,
                "height": 0
            },
            "bins": {
                "width": {
                    "x": [],  # the width
                    "y": []   # number of images with this width
                },
                "height": {
                    "x": [],  # the height
                    "y": []   # number of images with this height
                },
                "aspect_ratio": {
                    "<1": {
                        "x": [],
                        "y": []
                    },
                    "=1": {
                        "x": [],
                        "y": []
                    },
                    ">1": {
                        "x": [],
                        "y": []
                    }
                },
                "sqrt(h_w)": {
                    "x": [],
                    "y": []
                }
            },
        },
        "bbox": {
            "min": {
                "width": 65535,
                "height": 65535
            },
            "max": {
                "width": 0,
                "height": 0
            },
            "avg": {
                "width": 0,
                "height": 0
            },
            "bins": {
                "width": {
                    "x": [],  # the width
                    "y": []   # number of bboxes with this width
                },
                "height": {
                    "x": [],  # the height
                    "y": []   # number of bboxes with this height
                },
                "aspect_ratio": {
                    "<1": {
                        "x": [],
                        "y": []
                    },
                    "=1": {
                        "x": [],
                        "y": []
                    },
                    ">1": {
                        "x": [],
                        "y": []
                    }
                },
                "sqrt(h_w)": {
                    "x": [],
                    "y": []
                }
            },
        },
        "label_freq": {
            "min": 65535,     # minimal number of labels in an image
            "max": 0,         # maximum number of labels in an image
            "avg": 0,         # average number of labels in an image
            "bins": {
                "freq": {
                    "x": [i for i in range(1, num_cat + 1)],  # the label, resp. id of the label
                    "y": [0] * num_cat                        # frequency of the label in whole dataset
                },
                "avg": {
                    "x": [i for i in range(1, num_cat + 1)],  # the label, resp. id of the label
                    "y": [0] * num_cat                        # average frequency of the label in image
                }
            }
        }
    }

    # local functions
    def _rec_avg(act, x_n, n):
        # calculation of recursive average
        return (act * (n - 1) + x_n) / n

    def _get_stats(main_label, orientation_label, num_anns, img_width_or_height=1):
        if main_label == "image":
            if orientation_label not in ["width", "height"]:
                raise Exception("Error: Wrong 'orientation_label' is set... Expected one of [width, height], actual is "
                                f"'{orientation_label}'")
            orientation_value = img[orientation_label]
        elif main_label == "bbox":
            # COCO format of bounding box is [x, y, width, height]
            if orientation_label == "width":
                orientation_value = coco_data["annotations"][num_anns - 1]["bbox"][2] / img_width_or_height
            elif orientation_label == "height":
                orientation_value = coco_data["annotations"][num_anns - 1]["bbox"][3] / img_width_or_height
            else:
                raise Exception("Error: Wrong 'orientation_label' is set... Expected one of [width, height], actual is "
                                f"'{orientation_label}'")
        else:
            raise Exception("Error: Wrong 'main_label' is set... Expected one of [image, bbox], actual is "
                            f"'{main_label}'")

        # check min and max value
        if orientation_value < stats[main_label]["min"][orientation_label]:
            stats[main_label]["min"][orientation_label] = orientation_value
        if orientation_value > stats[main_label]["max"][orientation_label]:
            stats[main_label]["max"][orientation_label] = orientation_value

        # compute an average value
        stats[main_label]["avg"][orientation_label] = _rec_avg(
            stats[main_label]["avg"][orientation_label], orientation_value, num_anns
        )

        # save the bin
        if orientation_value not in stats[main_label]["bins"][orientation_label]["x"]:
            # append new value (of width/height)
            stats[main_label]["bins"][orientation_label]["x"].append(orientation_value)
            stats[main_label]["bins"][orientation_label]["y"].append(1)
        else:
            # search for the index and increase counter
            bin_index = stats[main_label]["bins"][orientation_label]["x"].index(orientation_value)
            stats[main_label]["bins"][orientation_label]["y"][bin_index] += 1

    def _sort_bins():
        for b in [stats["image"]["bins"]["width"], stats["image"]["bins"]["height"],
                  stats["bbox"]["bins"]["width"], stats["bbox"]["bins"]["height"],
                  stats["bbox"]["bins"]["aspect_ratio"]["<1"], stats["bbox"]["bins"]["aspect_ratio"]["=1"],
                  stats["bbox"]["bins"]["aspect_ratio"][">1"], stats["bbox"]["bins"]["sqrt(h_w)"]]:
            x, y = zip(*sorted(zip(
                b["x"], b["y"]
            )))

            b["x"] = list(x)
            b["y"] = list(y)

    # data processing
    ann_id = 1
    num_imgs = len(coco_data["images"])
    print(f"Getting {set_name} statistics..." if set_name is not None else "Getting dataset statistics...")
    time = datetime(2020, 1, 1)

    # init bins
    for t in ["<1", ">1"]:
        if t == "<1":
            stats["bbox"]["bins"]["aspect_ratio"][t]["x"] = [round(0.01 * i, 2) for i in range(0, 100)]
        else:
            stats["bbox"]["bins"]["aspect_ratio"][t]["x"] = [round(1.01 + 0.01 * i, 2) for i in range(0, 20000)]
        stats["bbox"]["bins"]["aspect_ratio"][t]["y"] = [0] * len(stats["bbox"]["bins"]["aspect_ratio"][t]["x"])

    stats["bbox"]["bins"]["aspect_ratio"]["=1"]["x"] = [1.00]
    stats["bbox"]["bins"]["aspect_ratio"]["=1"]["y"] = [0]

    stats["bbox"]["bins"]["sqrt(h_w)"]["x"] = [10 * i for i in range(0, 300)]
    stats["bbox"]["bins"]["sqrt(h_w)"]["y"] = [0] * len(stats["bbox"]["bins"]["sqrt(h_w)"]["x"])

    for img_id, img in enumerate(coco_data["images"], 1):
        # print status of processing
        time_now = datetime.now()
        if (time_now - time) > CONFIG["print_status"]:
            print(f"{img_id}/{num_imgs}")
            time = time_now

        # get statistics
        _get_stats("image", "width", img_id)
        _get_stats("image", "height", img_id)

        lbl_freq = 0
        lbl_freq_list = [0] * num_cat
        while (ann_id <= num_ann) and (coco_data["annotations"][ann_id - 1]["image_id"] == img_id):
            if relative:
                _get_stats("bbox", "width", ann_id, img["width"])
                _get_stats("bbox", "height", ann_id, img["height"])
            else:
                _get_stats("bbox", "width", ann_id)
                _get_stats("bbox", "height", ann_id)

            # get aspect ratio
            bbox_w = coco_data["annotations"][ann_id - 1]["bbox"][2]
            bbox_h = coco_data["annotations"][ann_id - 1]["bbox"][3]
            aspect_ratio = round(bbox_h / (bbox_w + 0.00000000001), 2)

            for type in ["<1", "=1", ">1"]:
                if type == "<1" and aspect_ratio >= 1:
                    continue
                elif type == ">1" and aspect_ratio <= 1:
                    continue
                elif type == "=1" and aspect_ratio != 1:
                    continue

                # search for the index and increase counter
                bin_index = stats["bbox"]["bins"]["aspect_ratio"][type]["x"].index(round(aspect_ratio, 2))
                stats["bbox"]["bins"]["aspect_ratio"][type]["y"][bin_index] += 1

            # sizes
            w_h = coco_data["annotations"][ann_id - 1]["bbox"][2] * coco_data["annotations"][ann_id - 1]["bbox"][3]
            size = int(round(math.sqrt(w_h), -1))
            if size not in stats["bbox"]["bins"]["sqrt(h_w)"]["x"]:
                # append new value
                stats["bbox"]["bins"]["sqrt(h_w)"]["x"].append(size)
                stats["bbox"]["bins"]["sqrt(h_w)"]["y"].append(1)
            else:
                # search for the index and increase counter
                bin_index = stats["bbox"]["bins"]["sqrt(h_w)"]["x"].index(size)
                stats["bbox"]["bins"]["sqrt(h_w)"]["y"][bin_index] += 1

            # actualize label frequency
            cat_id = coco_data["annotations"][ann_id - 1]["category_id"]
            stats["label_freq"]["bins"]["freq"]["y"][cat_id - 1] += 1
            lbl_freq_list[cat_id - 1] += 1  # used for average freq per image

            # get next annotation
            lbl_freq += 1
            ann_id += 1

        # actualize remaining statistics
        if lbl_freq < stats["label_freq"]["min"]:
            stats["label_freq"]["min"] = lbl_freq
        if lbl_freq > stats["label_freq"]["max"]:
            stats["label_freq"]["max"] = lbl_freq
        stats["label_freq"]["avg"] = _rec_avg(
            stats["label_freq"]["avg"], lbl_freq, img_id
        )

        # compute average frequency per image
        lbl_avg = stats["label_freq"]["bins"]["avg"]["y"]
        stats["label_freq"]["bins"]["avg"]["y"] = [_rec_avg(lbl_avg[i], lbl_freq_list[i], img_id)
                                                   for i in range(num_cat)]

    # sort all bins
    _sort_bins()

    non_zero_asprat = [i for i, e in enumerate(stats["bbox"]["bins"]["aspect_ratio"][">1"]["y"]) if e > 0]
    lst_asprat = non_zero_asprat[-1]
    for t in ["<1", "=1", ">1"]:
        if t == ">1":
            stats["bbox"]["bins"]["aspect_ratio"][t]["x"] = stats["bbox"]["bins"]["aspect_ratio"][t]["x"][:lst_asprat]
            stats["bbox"]["bins"]["aspect_ratio"][t]["y"] = stats["bbox"]["bins"]["aspect_ratio"][t]["y"][:lst_asprat]

        # to have probabilities
        stats["bbox"]["bins"]["aspect_ratio"][t]["y"] = [i / ann_id
                                                         for i in stats["bbox"]["bins"]["aspect_ratio"][t]["y"]]

    stats["bbox"]["bins"]["sqrt(h_w)"]["x"] = stats["bbox"]["bins"]["sqrt(h_w)"]["x"][:100]
    stats["bbox"]["bins"]["sqrt(h_w)"]["y"] = stats["bbox"]["bins"]["sqrt(h_w)"]["y"][:100]

    # to have probabilities
    stats["bbox"]["bins"]["sqrt(h_w)"]["y"] = [i / ann_id
                                               for i in stats["bbox"]["bins"]["sqrt(h_w)"]["y"]]

    # save data, if required
    if set_name is not None:
        output_path = os.path.join(CONFIG["output_path"], "get_stats")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = os.path.join(output_path, f"stats_{set_name}.json")
        with open(output_path, "w") as output_file:
            json.dump(stats, output_file)
            print(f"Statistics of dataset file saved: {output_path}")

    print("")

    return stats


def vis_stats(stats_data, set_name=None, show=False, save=True):
    def _print(string):
        print(string)
        if save is True and output_file is not None:
            output_file.write(f"{string}\n")

    # set output path
    output_path = None
    output_file = None
    if save:
        output_path = os.path.join(CONFIG["output_path"], "vis_stats")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file = open(os.path.join(output_path, f"{set_name}_log.txt"), "w")

    # print status
    if set_name is not None:
        _print("=" * 50)
        _print(f"Stats of dataset: {set_name}")
        _print("=" * 50)
    else:
        set_name = ""

    # w = what?
    for w in ["image", "bbox"]:
        # b = bin; l = label
        for b, l in [(stats_data[w]["bins"]["width"], "width"),
                     (stats_data[w]["bins"]["height"], "height"),
                     (stats_data[w]["bins"]["aspect_ratio"], "aspect_ratio"),
                     (stats_data[w]["bins"]["sqrt(h_w)"], "sqrt(h_w)")]:
            if (l == "aspect_ratio" and w != "bbox") or (l == "sqrt(h_w)" and w != "bbox"):
                continue
            elif l == "aspect_ratio" and w == "bbox":
                b = {
                    "x": b["<1"]["x"] + b["=1"]["x"] + b[">1"]["x"][:100],
                    "y": b["<1"]["y"] + b["=1"]["y"] + b[">1"]["y"][:100]
                }

            # set x as list from 0 to number of values, it is used as new x axis for plotting
            x = [i for i in range(len(b["x"]))]
            max_i = len(x) - 1
            num_ticks = 21

            # get original ticks and ticks used for plotting
            if isinstance(b["x"][0], int):
                is_float = False
                ticks_plot = [x[int(max_i * i / (num_ticks - 1))] for i in range(num_ticks)]
                ticks_orig = [b["x"][int(max_i * i / (num_ticks - 1))] for i in range(num_ticks)]
            else:
                is_float = True
                ticks_plot = [round(x[int(max_i * i / (num_ticks - 1))], 3) for i in range(num_ticks)]
                ticks_orig = [round(b["x"][int(max_i * i / (num_ticks - 1))], 3) for i in range(num_ticks)]

            # plot an figure
            fig = plt.figure(figsize=(15, 5))
            plt.bar(x, b["y"])
            plt.title(f'Occurrence of {w} {l}s in {set_name}')
            plt.xticks(ticks_plot, ticks_orig)
            plt.xlabel(l)
            plt.ylabel(f'probability')
            if show:
                plt.show()
            if save:
                fig.savefig(os.path.join(output_path, f"{set_name}_{w}_{l}.eps"), format='eps')

            # close plot / save memory
            plt.close()

            # print statistics
            y, x = zip(*sorted(zip(
                b["y"], b["x"]
            )))

            if is_float:
                top_10 = [f"%.3f/%d" % (x[-i], y[-i]) for i in range(1, 11)]
            else:
                top_10 = [f"%d/%d" % (x[-i], y[-i]) for i in range(1, 11)]
            if l != "aspect_ratio" and l != "sqrt(h_w)":
                s = [stats_data[w]['avg'][l], stats_data[w]['min'][l], stats_data[w]['max'][l]]
                _print(f"[Avg, Min, Max] {w} {l}: {s}")
            _print(f"Top 10 {w} {l}s ({l}/cnt): {top_10}")
            _print("")

    for b, l in [(stats_data["label_freq"]["bins"]["freq"], "freq"),
                 (stats_data["label_freq"]["bins"]["avg"], "avg")]:
        # plot label frequency
        fig = plt.figure()
        plt.bar(b["x"], b["y"])
        if l == "freq":
            plt.title(f'Label frequency in {set_name}.')
        elif l == "avg":
            plt.title(f'Average label frequency in {set_name}.')
        plt.xticks(b["x"], b["x"])
        plt.xlabel('category ID')
        plt.ylabel('occurrences')
        if show:
            plt.show()
        if set_name is not None:
            fig.savefig(os.path.join(output_path, f"{set_name}_label_{l}.png"))

        # close plot / save memory
        plt.close()

    # print statistics about label frequency
    s = [stats_data["label_freq"]['avg'], stats_data["label_freq"]['min'], stats_data["label_freq"]['max']]
    _print(f"[Avg, Min, Max] number of labels: {s}")
    _print("=" * 50)
    _print("")

    if save:
        output_file.close()


def make_submission_file(eval_json, coco_data, output_dir=""):
    submit = list()
    with open(eval_json, "r") as json_file:
        eval = json.load(json_file)

        # get dicts by id
        coco_imgs = dict()
        for img in coco_data["images"]:
            coco_imgs[int(img["id"])] = [img["file_name"], img["width"], img["height"]]
        FILE_NAME, WIDTH, HEIGHT = 0, 1, 2

        coco_cats = dict()
        for cat in coco_data["categories"]:
            coco_cats[int(cat["id"])] = cat["name"]

        # prepare submission file
        annots = list()
        img_id_last = eval[0]["image_id"]  # initial image_id
        for idx in range(len(eval)):
            # {"image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9863572120666504},
            img_id = eval[idx]["image_id"]
            if img_id != img_id_last:
                if len(annots) > 300:
                    annots = annots[:300]

                submit.append({
                    "file": coco_imgs[img_id_last][FILE_NAME],
                    "width": coco_imgs[img_id_last][WIDTH],
                    "height": coco_imgs[img_id_last][HEIGHT],
                    "annotations": annots
                })
                img_id_last = img_id
                annots = list()

            annots.append({
                "score": eval[idx]["score"],
                "detectionClass": eval[idx]["category_id"],
                "detectionString": coco_cats[eval[idx]["category_id"]],
                "box": list(encode_bbox(eval[idx]["bbox"], coco_imgs[img_id][WIDTH], coco_imgs[img_id][HEIGHT]))
            })

        if len(annots) != 0:
            if len(annots) > 300:
                annots = annots[:300]

            submit.append({
                "file": coco_imgs[img_id_last][FILE_NAME],
                "width": coco_imgs[img_id_last][WIDTH],
                "height": coco_imgs[img_id_last][HEIGHT],
                "annotations": annots
            })

    output_path = output_dir if output_dir != "" else os.path.join(CONFIG["output_path"], "make_submission_file")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, "submission.json"), "w") as output_file:
        json.dump(submit, output_file)


if __name__ == "__main__":
    # Parse commandline
    parser = argparse.ArgumentParser(description='Dataset tools for competition "ImageCLEF DrawnUI 2021".')

    # Optional arguments
    parser.add_argument('-st', '--ss_train', action='store_true', help='Process "Screenshot" training data.')
    parser.add_argument('-sv', '--ss_valid', action='store_true', help='Process "Screenshot" validation data.')
    parser.add_argument('-wd', '--wf_dev', action='store_true', help='Process "Wireframe" development data.')

    parser.add_argument('-D', '--vis_data', action='store_true', help='Visualize the data images and store them.')
    parser.add_argument('-S', '--vis_stats', action='store_true', help='Visualize the data stats and store them.')
    parser.add_argument('-C', '--make_coco', action='store_true', help='Make and store annotations in COCO format.')
    parser.add_argument('-L', '--vis_learning_progress', action='store_true', help='Visualize the learning progress.'
                                                                                   ' --log_file parameter is required!')
    parser.add_argument('-E', '--make_submission_file', action='store_true', help='Make file for submission.')

    parser.add_argument('-l', '--log_file', default="", help='Path to the log file that outputs DETR after training.')
    parser.add_argument('-e', '--eval_file', default="", help='Path to the eval file that outputs DETR after training.')

    # Parsing arguments
    args = parser.parse_args()

    # =================================================================================================================
    # check output path
    if not os.path.exists(CONFIG["output_path"]):
        print(f"Making dirs... {CONFIG['output_path']}")
        os.makedirs(CONFIG["output_path"])

    # define data, which would be processed
    process_data = list()  # [original_json, original_data, dataset_name, split_data]
    if args.ss_train:
        process_data.append(
            [CONFIG["screenshot_train_json"], CONFIG["screenshot_train_data"], "screenshot_train", "false"]
        )
    if args.ss_valid:
        process_data.append(
            [CONFIG["screenshot_valid_json"], CONFIG["screenshot_valid_data"], "screenshot_valid", "false"]
        )
    if args.wf_dev:
        process_data.append(
            [CONFIG["wireframe_json"], CONFIG["wireframe_data"], "wireframe", "true"]
        )

    # =================================================================================================================
    # call required functions
    stats = {}
    coco = {}
    for json_data, img_data, dataset_name, split_data in process_data:
        if args.vis_data:
            vis_data("D:/drawnUI2021_experimenty_new/exp6_backbones/wireframe/exp6_011_e14_faster_rcnn_X_101_32x8d_FPN_3x_lr_0.0025_b_1_a_4_fc_256_ff_sum_bc_0_bf_2_bcd_256_bfd_1024_d_320_e_14/submission/submission.json", "D:/ImageCLEFdrawnUI2021/data/wireframe_test_set/test", show=False, save=True, set_name="detr_output_wf")
            #vis_data(json_data, img_data, show=False, save=True, set_name=dataset_name)

        if args.make_coco:
            coco = make_coco(json_data, dataset_name, img_data, split_data, reproducibility_set=False
                             , discard=["outlier_boxes", "hmg_boxes", "hmg_imgs"])
            if len(coco) == 2:
                for i, dataset_type in enumerate(["train", "valid"]):
                    copy_data(coco[i], img_data, f"{dataset_name}_{dataset_type}")

        if args.vis_stats:
            if not args.make_coco:
                coco = make_coco(json_data)

            if len(coco) == 1:
                stats = get_stats(coco[0], dataset_name)
                vis_stats(stats, set_name=dataset_name, show=False, save=True)
            else:
                for i, dataset_type in enumerate(["train", "valid"]):
                    stats = get_stats(coco[i], f"{dataset_name}_{dataset_type}")
                    vis_stats(stats, set_name=f"{dataset_name}_{dataset_type}", show=False, save=True)

        if args.vis_learning_progress:
            if args.log_file != "":
                vis_learning_progress(args.log_file, CONFIG["plt_keys"], show=False, save=True)
            else:
                print("Warning: argument --log_file was not set, learning progress cannot be visualized.")

        if args.make_submission_file:
            coco = make_coco(json_data, dataset_name, img_data, split_data)
            if args.eval_file != "":
                # TODO: zatím funguje jen pro validační data!!!!!!!! předávat ještě další parametr na coco
                make_submission_file(args.eval_file, coco[1])
            else:
                print("Warning: argument --eval_file was not set, submission file cannot be created.")
