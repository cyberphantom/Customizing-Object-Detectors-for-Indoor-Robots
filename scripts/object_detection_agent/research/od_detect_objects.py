import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from matplotlib import pyplot as plt
import skimage.io

# Env setup
# # This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
OBJECT_DETECTION_PATH = '/home/saif/catkin_ws/src/ucf_ardrone_ros/scripts/object_detection_agent/research/object_detection/'

'''Path to object detection folder'''
sys.path.append("/home/saif/catkin_ws/src/ucf_ardrone_ros/scripts/object_detection_agent/research")
sys.path.append("/home/saif/catkin_ws/src/ucf_ardrone_ros/scripts/object_detection_agent/research")

# Object detection imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Variables
# What model to download.
MODEL_NAME = '/ssd_mobilenet_v1_coco_11_06_2017/'
#MODEL_NAME = 'saif_pets'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(OBJECT_DETECTION_PATH, 'model_zoo' + MODEL_NAME + 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_PATH, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = os.path.join(OBJECT_DETECTION_PATH, 'test_images')
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)




class DetectImage(object):
    detection_graph = 'graph'
    sess = 'sess'
    image_tensor = 'Tensor'
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = 'Tensor'
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = 'Tensor'
    detection_classes = 'Tensor'
    num_detections = 'Tensor'

    def __init__(self):
        # Definite and open graph and sess
        self.detection_graph = detection_graph
        self.sess = tf.Session(graph=detection_graph)
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess.close()

    def run_detect(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        return image_np


if __name__ == '__main__':
    di = DetectImage()

    PATH_TO_TEST_IMAGES_DIR = os.path.join(OBJECT_DETECTION_PATH, 'test_images')
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
    for image_path in TEST_IMAGE_PATHS:
        # image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # image_np = load_image_into_numpy_array(image)
        image_np = skimage.io.imread(image_path)
        image_np = di.run_detect(image_np)

        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.show()