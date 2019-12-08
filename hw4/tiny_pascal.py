"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
"""
import matplotlib.pyplot as plt
import cv2
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("D:/user/Desktop/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class TinyPascalConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tiny_pascal"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 20  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class TinyPascalDataset(utils.Dataset):

    def load_tiny_pascal(self, dataset_dir, subset):
        ann_dir = os.path.join(dataset_dir, "pascal_train.json")
        dataSet = COCO(ann_dir)

        image_dir = os.path.join(dataset_dir, subset)

        # Get the class id 
        class_ids = sorted(dataSet.getCatIds())
        #print(class_ids)
        
        # Get the imgs id
        image_ids = list(dataSet.imgs.keys())
        #print(image_ids[:20])

        # Add classes
        for i in class_ids:
            self.add_class("tiny_pascal", i, dataSet.loadCats(i)[0]["name"])
        
        # Add images
        for i in image_ids:
            self.add_image(
                "tiny_pascal", image_id=i,
                path=os.path.join(image_dir, dataSet.imgs[i]['file_name']),
                width=dataSet.imgs[i]["width"],
                height=dataSet.imgs[i]["height"],
                annotations=dataSet.loadAnns(dataSet.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a tiny_pascal image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "tiny_pascal":
           return super(TinyPascalDataset, self).load_mask(image_id)
        
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "tiny_pascal.{}".format(annotation['category_id']))
            print(class_id)
        return instance_masks, class_ids

if __name__ == '__main__':
    dataset_tiny_pascal = TinyPascalDataset()
    dataset_tiny_pascal.load_tiny_pascal("D:/user/Desktop/DLHW4/", "train_images")
    dataset_tiny_pascal.prepare()
    
    print("Image Count: {}".format(len(dataset_tiny_pascal.image_ids)))
    print("Class Count: {}".format(dataset_tiny_pascal.num_classes))
    
    image = dataset_tiny_pascal.load_image(1)
    plt.imshow(image)
    plt.savefig("D:/user/Desktop/DLHW4/image.png")
    mask, class_ids = dataset_tiny_pascal.load_mask(1)

