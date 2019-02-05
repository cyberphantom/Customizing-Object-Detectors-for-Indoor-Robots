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
MODEL1_NAME = '/ssd_mobilenet_v1_coco_11_06_2017/'
MODEL2_NAME = '/fcrar18_mix_checkpoints/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT1 = os.path.join(OBJECT_DETECTION_PATH, 'model_zoo' + MODEL1_NAME + 'frozen_inference_graph.pb')
PATH_TO_CKPT2 = os.path.join(OBJECT_DETECTION_PATH, 'model_zoo' + MODEL2_NAME + '/frozen_inference_graph.pb')


# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_PATH, 'data', 'object-detection.pbtxt')
PATH_TO_LABELS = os.path.join(OBJECT_DETECTION_PATH, 'data', 'mix.pbtxt')

NUM_CLASSES = 94


# Load a (frozen) Tensorflow model into memory.
detection_graph1 = tf.Graph()
with tf.device("/gpu:0"):
    with detection_graph1.as_default():
        od_graph_def1 = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT1, 'rb') as fid1:
            serialized_graph1 = fid1.read()
            od_graph_def1.ParseFromString(serialized_graph1)
            tf.import_graph_def(od_graph_def1, name='')


detection_graph2 = tf.Graph()
with tf.device("/gpu:1"):
    with detection_graph2.as_default():
        od_graph_def2 = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT2, 'rb') as fid2:
            serialized_graph2 = fid2.read()
            od_graph_def2.ParseFromString(serialized_graph2)
            tf.import_graph_def(od_graph_def2, name='')

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


class DetectMix(object):
    detection_graph1 = 'graph'
    detection_graph2 = 'graph'
    sess1 = 'sess'
    sess2 = 'sess'
    image_tensor1 = 'Tensor'
    image_tensor2 = 'Tensor'
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes1 = 'Tensor'
    detection_boxes2 = 'Tensor'
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores1 = 'Tensor'
    detection_scores2 = 'Tensor'
    detection_classes1 = 'Tensor'
    detection_classes2 = 'Tensor'
    num_detections1 = 'Tensor'
    num_detections2 = 'Tensor'

    def __init__(self):
        # Definite and open graph and sess
        self.detection_graph1 = detection_graph1
        self.sess1 = tf.Session(graph=detection_graph1, config=tf.ConfigProto(log_device_placement=True))

        self.detection_graph2 = detection_graph2
        self.sess2 = tf.Session(graph=detection_graph2, config=tf.ConfigProto(log_device_placement=True))

        # Definite input and output Tensors for detection_graph
        self.image_tensor1 = self.detection_graph1.get_tensor_by_name('image_tensor:0')
        self.image_tensor2 = self.detection_graph2.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes1 = self.detection_graph1.get_tensor_by_name('detection_boxes:0')
        self.detection_boxes2 = self.detection_graph2.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores1 = self.detection_graph1.get_tensor_by_name('detection_scores:0')
        self.detection_classes1 = self.detection_graph1.get_tensor_by_name('detection_classes:0')
        self.num_detections1 = self.detection_graph1.get_tensor_by_name('num_detections:0')

        self.detection_scores2 = self.detection_graph2.get_tensor_by_name('detection_scores:0')
        self.detection_classes2 = self.detection_graph2.get_tensor_by_name('detection_classes:0')
        self.num_detections2 = self.detection_graph2.get_tensor_by_name('num_detections:0')

    def __del__(self):
        self.sess1.close()
        self.sess2.close()

    def run_detect(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes1, scores1, classes1, num1) = self.sess1.run(
            [self.detection_boxes1, self.detection_scores1, self.detection_classes1, self.num_detections1],
            feed_dict={self.image_tensor1: image_np_expanded})

        (boxes2, scores2, classes2, num2) = self.sess2.run(
            [self.detection_boxes2, self.detection_scores2, self.detection_classes2, self.num_detections2],
            feed_dict={self.image_tensor2: image_np_expanded})
        # Visualization of the results of a detection.





        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes1),
            np.squeeze(classes1).astype(np.int32),
            np.squeeze(scores1),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes2),
            np.squeeze(classes2).astype(np.int32),
            np.squeeze(scores2),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return image_np


if __name__ == '__main__':
    di = DetectMix()

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
