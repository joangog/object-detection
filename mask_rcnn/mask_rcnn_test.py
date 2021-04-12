import os
import time
import tensorflow as tf
import numpy as np
import cv2
import skimage

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from mrcnn.model import MaskRCNN
from mask_rcnn.assets.visualize import display_instances
from mask_rcnn.assets.mask_rcnn_config import MaskConfig, MaskDataset


## Testing Functions ###################################################################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    # Arrange results to match COCO specs in http://cocodataset.org/#format

    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "masks"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

def test_model(model, dataset, annotations, eval_type="bbox"):

    image_ids = dataset.image_ids

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = annotations.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(annotations, coco_results, eval_type)
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

########################################################################################################################

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))  # Detect GPU

# Directory of project
root_dir = os.path.abspath("../")

# Directory of dataset
dataset_dir = os.path.join(root_dir, "dataset")

# Directory of model
model_dir = os.path.join(root_dir, "mask_rcnn")

# Directory of model assets
assets_dir = os.path.join(model_dir, "assets")

# File name
input_file = 'mask1.jpg'
output_file = 'pedestrians_detected.mp4'



# # Load video
# video = cv2.VideoCapture(os.path.join(image_dir, input_file))
# frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # approximate frame count
#
# print("Total frames to render: " + str(frame_count))

# # Initialize video writer
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# writer = cv2.VideoWriter(os.path.join(image_dir, output_file), fourcc, 30, (frame_width, frame_height), True)

# Define test dataset
dataset_test = MaskDataset()
annotations = dataset_test.load_dataset(dataset_dir, 'test', return_ann=True)
dataset_test.prepare()

# Define model configurations
config = MaskConfig()

# Define model
model = MaskRCNN(mode='inference', model_dir='./log', config=config)

# Load pre-trained COCO weights
model.load_weights(os.path.join(assets_dir, "mask_rcnn_masks_0016.h5"), by_name=True)

# Run testing
test_model(model, dataset_test, annotations, "bbox")

# Render masked image
# masked_frame = display_instances(dataset_train, results[0]['rois'], results[0]['masks'], results[0]['class_ids'],
#                             class_names, results[0]['scores'])

# cv2.imshow('result', masked_frame)  # Show rendered image
# cv2.waitKey(0)

