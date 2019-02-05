from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import cv2, os
from dronet import dronenet
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms


print (os.path.join('home/saif/Documents/datasets/droneSet/weights/myTrain/dronenet_fpn_original_augment' + 'dronenet_epoch-136_loss-3.4134_val_loss-3.0895.h5'))