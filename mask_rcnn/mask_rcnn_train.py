import os
import tensorflow as tf

from mrcnn.model import MaskRCNN
from mask_rcnn.scripts.mask_rcnn_config import MaskConfig, MaskDataset


sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))  # Detect GPU

# Directory of project
root_dir = os.path.abspath("../")

# Directory of dataset
dataset_dir = os.path.join(root_dir, "dataset")

# Directory of model
model_dir = os.path.join(root_dir, "mask_rcnn")

# Directory of model assets
assets_dir = os.path.join(model_dir, "assets")

# Define model configurations
config = MaskConfig()

# Define dataset
dataset_train = MaskDataset()
dataset_train.load_dataset(dataset_dir, 'train')
dataset_train.prepare()
dataset_val = MaskDataset()
dataset_val.load_dataset(dataset_dir, 'val')
dataset_val.prepare()

# Define model
model = MaskRCNN(mode='training', model_dir='./log', config=config)
model.load_weights(os.path.join(assets_dir, "mask_rcnn_masks_0010.h5"), by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# Training
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='heads')