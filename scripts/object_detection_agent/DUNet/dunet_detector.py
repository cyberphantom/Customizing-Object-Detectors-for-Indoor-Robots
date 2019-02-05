from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam, SGD
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt
import cv2, os
import tensorflow as tf
from dronenet_fpn_256 import dronenet
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


class dronet_detect():

    def __init__(self):
        self.img_height = 320
        self.img_width = 320
        self.frame = None
        self.model = dronenet(image_size=(self.img_height, self.img_width, 3),
                n_classes=10,
                mode='inference',
                l2_regularization=0.0005,
                #scales=[0.035, 0.07, 0.15, 0.33, 0.62], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                #scales=[0.05, 0.2, 0.4, 0.6, 0.8],
                scales = [0.035, 0.075, 0.22, 0.45, 0.75], # 128
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0]],
                two_boxes_for_ar1=True,
                steps=[4, 8, 16, 32],
                offsets=[0.5, 0.5, 0.5, 0.5],
                clip_boxes=True,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5, # check
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

        #weights_path = os.path.join('/home/saif/Documents/datasets/droneSet/weights/myTrain/dronenet_fpn_scale/' + 'dronenet_epoch-160_loss-3.2399_val_loss-3.0114.h5')
        weights_path = os.path.join(
            '/home/saif/Documents/datasets/droneSet/weights/myTrain/dronenet_fpn_256_original_augment/' + 'dronenet_epoch-185_loss-3.7147_val_loss-3.5914.h5')
        self.model.load_weights(weights_path, by_name=True)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        self.model.compile(optimizer=sgd, loss=ssd_loss.compute_loss)

        self.graph = tf.get_default_graph()
        self.color = list(np.random.choice(range(0, 256, 50), size=3))

        self.classes = ['background',
                   'Christmas toy', 'coffee machine', 'potted plant', 'tissue box',
                   'robot', 'soccer ball', 'turtle bot', 'uav',
                   'fire alarm', 'tennis racket']


    def predictor(self, frame):
        self.frame= None
        info = []
        with self.graph.as_default():
            img = cv2.resize(frame, dsize=(self.img_height, self.img_width), interpolation=cv2.INTER_NEAREST)
            np_img = np.expand_dims(img, axis=0)
            y_pred = self.model.predict(np_img)
            confidence_threshold = 0.5
            y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
            np.set_printoptions(precision=2, suppress=True, linewidth=90)

            for i, box in enumerate(y_pred_thresh[0]):
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = int(box[2] * frame.shape[1] / self.img_width)
                ymin = int(box[3] * frame.shape[0] / self.img_height)
                xmax = int(box[4] * frame.shape[1] / self.img_width)
                ymax = int(box[5] * frame.shape[0] / self.img_height)
                label = '{}: {:.2f}'.format(self.classes[int(box[0])], box[1])
                info.append(label)
                self.frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), self.color[i], thickness=2)
                self.frame = cv2.putText(self.frame, label, (xmin, ymin-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color[i])


        if self.frame is None:
             return frame, y_pred_thresh[0]
        else:
            return self.frame, info




