# -*- encoding: utf-8 -*-
# -------------------------------------------
# Year 4 personal project code work dataset loading part
# -------------------------------------------
# Zhengyu

import os
import json
import numpy as np
from options import *
from PIL import Image
import matplotlib.pyplot as plt


def prepare_data_folders(args):
    """
    # for preparing the dataset paths
    :return:
    """
    dataset_path = args.dataset_path
    # threat items: knife, firearm, firearmparts
    threat_items_path = dataset_path + "/threatItems"
    threat_id_dict = {1: "/firearm", 2: "/firearmparts", 3: "/knife"}
    args.threat_item = threat_id_dict[args.threat_id]
    threat_item_list = [os.path.join(threat_items_path + threat_id_dict[args.threat_id], item_path)
                                  for item_path in
                                  os.listdir(threat_items_path + threat_id_dict[args.threat_id]) if
                                  ".DS_Store" not in item_path]

    # background list 3366 images
    background_path_list = [os.path.join(dataset_path + "/negative", background_path) for background_path
                                      in
                                      os.listdir(dataset_path + "/negative") if ".DS_Store" not in background_path]
    # print(background_path_list)

    # real image path list and the annotation json data
    real_image_path = dataset_path + "/real_images"
    annotation_path = dataset_path + "/annotation"

    with open(annotation_path + "/dbf4_train.json", 'r') as load_f:
        annotation_json = json.load(load_f)
    real_image_path_train = real_image_path + "/train_set"

    # load the train real image path
    # adding the label of the threat items
    # {'id': 1, 'name': 'firearm'},
    # {'id': 2, 'name': 'firearmpart'},
    # {'id': 3, 'name': 'knife'},
    # {'id': 4, 'name': 'ceramicknife'}
    ann_infos = annotation_json["annotations"]
    exact_threat_an_infos = list(filter(lambda x_dict: x_dict["category_id"] == args.threat_id, ann_infos))
    exact_threat_an_infos = sorted(exact_threat_an_infos, key=lambda x: x.__getitem__("image_id"), reverse=True)
    # print(exact_threat_an_infos)
    binary_box_list = [list(map(int, threat_info["bbox"])) for threat_info in exact_threat_an_infos]
    segmentation_list = [list(map(int, threat_info["segmentation"][0])) for threat_info in
                                   exact_threat_an_infos]

    image_id_list = [threat_info["image_id"] for threat_info in exact_threat_an_infos]
    # print(image_id_list)
    image_infos = annotation_json["images"]
    image_infos = sorted(image_infos, key=lambda x: x.__getitem__("id"), reverse=True)
    real_image_name_list = [image_info["file_name"] for image_info in image_infos if
                            image_info["id"] in image_id_list]
    real_path_list = [os.path.join(real_image_path_train, real_path) for real_path in real_image_name_list if
                                ".DS_Store" not in real_path]

    with open(annotation_path + "/dbf4_test.json", 'r') as load_f:
        annotation_json = json.load(load_f)
    real_image_path_test = real_image_path + "/test_set"

    # load the test real image path
    # adding the label of the threat items
    # {'id': 1, 'name': 'firearm'},
    # {'id': 2, 'name': 'firearmpart'},
    # {'id': 3, 'name': 'knife'},
    # {'id': 4, 'name': 'ceramicknife'}
    ann_infos = annotation_json["annotations"]
    exact_threat_an_infos = list(filter(lambda x_dict: x_dict["category_id"] == args.threat_id, ann_infos))
    exact_threat_an_infos = sorted(exact_threat_an_infos, key=lambda x: x.__getitem__("image_id"), reverse=True)
    binary_box_list += [list(map(int, threat_info["bbox"])) for threat_info in exact_threat_an_infos]
    segmentation_list += [list(map(int, threat_info["segmentation"][0])) for threat_info in
                                    exact_threat_an_infos]

    image_id_list = [threat_info["image_id"] for threat_info in exact_threat_an_infos]
    image_infos = annotation_json["images"]
    image_infos = sorted(image_infos, key=lambda x: x.__getitem__("id"), reverse=True)
    real_image_name_list = [image_info["file_name"] for image_info in image_infos if
                            image_info["id"] in image_id_list]
    real_path_list += [os.path.join(real_image_path_test, real_path) for real_path in real_image_name_list if
                                 ".DS_Store" not in real_path]

    # print the number of the datasets that loaded
    print("Threat image number:", len(threat_item_list))
    print("Background image number:", len(background_path_list))
    print("Real image number:", len(real_path_list))

    return threat_item_list, background_path_list, real_path_list


if __name__ == "__main__":
    """
    # For function testing
    """
    args = Options().parse()

    threat_item_list, background_path_list, real_path_list = prepare_data_folders(args)

    threat_example = Image.open(threat_item_list[20])
    plt.imshow(threat_example)
    plt.show()

    bg_example = Image.open(background_path_list[9])
    plt.imshow(bg_example)
    plt.show()

    # image of the real image ()
    index = 20
    real_example = Image.open(real_path_list[200])
    plt.imshow(real_example)
    plt.show()
