import os, sys
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

    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256
    MINI_MASK_SHAPE = (128, 128)

    STEPS_PER_EPOCH = 100
    LEARNING_RATE = 0.002
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

class MaskDataset(Dataset):

    def load_dataset(self, dataset_dir, type):  # modified function from coco.py of mrcnn library

        annotation_dir = os.path.join(dataset_dir, type+".json")
        image_dir = os.path.join(dataset_dir, type+"_images")

        annotations = COCO(annotation_dir)
        class_ids = sorted(annotations.getCatIds())
        image_ids = list(annotations.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("mask-detection", i, annotations.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "mask-detection", image_id=i,
                path=os.path.join(image_dir, annotations.imgs[i]['file_name']),
                width=annotations.imgs[i]["width"],
                height=annotations.imgs[i]["height"],
                annotations=annotations.loadAnns(annotations.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):  # modified function from coco.py of mrcnn library

        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        for annotation in annotations:
            class_id = self.map_source_class_id("mask-detection.{}".format(annotation['category_id']))
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