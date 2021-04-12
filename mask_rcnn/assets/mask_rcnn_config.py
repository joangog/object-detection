import os, sys
import time
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from mrcnn.config import Config
from mrcnn.utils import Dataset

class MaskConfig(Config):
    NAME = "masks"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # 2 classes + 1 for background

    IMAGE_CHANNEL_COUNT = 3
    IMAGE_META_SIZE = 15
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256
    IMAGE_MIN_SCALE = 0
    IMAGE_RESIZE_MODE = 'square'
    IMAGE_SHAPE = [256, 256, 3]

    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

    BACKBONE = 'resnet50'
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    BBOX_STD_DEV = [0.1,  0.1, 0.2, 0.2]
    COMPUTE_BACKBONE_SHAPE = None
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE =  0.75
    DETECTION_NMS_THRESHOLD = 0.3
    FPN_CLASSIF_FC_LAYERS_SIZE =  1024
    GRADIENT_CLIP_NORM = 5.0
    LOSS_WEIGHTS = {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}

    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    MINI_MASK_SHAPE = (128, 128)

    MAX_GT_INSTANCES = 35
    MEAN_PIXEL = [123.7,  116.8, 103.9]
    POOL_SIZE = 7
    POST_NMS_ROIS_INFERENCE = 1000
    POST_NMS_ROIS_TRAINING = 2000
    PRE_NMS_LIMIT = 6000
    ROI_POSITIVE_RATIO = 0.33
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE = 1
    RPN_BBOX_STD_DEV = [0.1,  0.1,  0.2, 0.2]
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    STEPS_PER_EPOCH = 150
    TOP_DOWN_PYRAMID_SIZE = 256
    TRAIN_BN = False
    TRAIN_ROIS_PER_IMAGE = 200
    USE_MINI_MASK = True
    USE_RPN_ROIS = True
    VALIDATION_STEPS = 50

class MaskDataset(Dataset):  # modified CocoDataset from coco.py of mrcnn library

    def load_dataset(self, dataset_dir, type, return_ann=False):

        annotation_dir = os.path.join(dataset_dir, type+".json")
        image_dir = os.path.join(dataset_dir, type+"_images")

        annotations = COCO(annotation_dir)
        class_ids = sorted(annotations.getCatIds())
        image_ids = list(annotations.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("masks", i, annotations.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "masks", image_id=i,
                path=os.path.join(image_dir, annotations.imgs[i]['file_name']),
                width=annotations.imgs[i]["width"],
                height=annotations.imgs[i]["height"],
                annotations=annotations.loadAnns(annotations.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

        if return_ann:
            return annotations


    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id("masks.{}".format(annotation['category_id']))
            if class_id:
                rles = maskUtils.frPyObjects(annotation['segmentation'], image_info['height'], image_info['width'])
                rle = maskUtils.merge(rles) # convert to RLE format
                m = maskUtils.decode(rle)  # decode RLE format to mask
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["id"]

