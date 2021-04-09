import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide TensorFlow warnings
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

from yolov4.tf import YOLOv4

# Directory of project
root_dir = os.path.abspath("../")
#sys.path.append(root_dir)  # To find local version of the library

# Directory of ataset
dataset_dir = os.path.join(root_dir, "dataset")

# Directory of model
model_dir = os.path.join(root_dir, "yolo")

# Directory of model assets
assets_dir = os.path.join(model_dir, "assets")

# Define model
model = YOLOv4()

# Define model configurations
model.config.parse_cfg(os.path.join(assets_dir, "yolov4_coco.cfg"))

model.make_model()

# Load pre-trained COCO weights

model.load_weights(os.path.join(assets_dir, "yolov4_coco.weights"), weights_type="yolo")

# Define COCO class names
model.config.parse_names(os.path.join(assets_dir, "coco.names"))

model.summary(summary_type="yolo")
model.summary()

# Run detection on video stream from camera (press q to stop)
model.inference(0,is_image=False)
