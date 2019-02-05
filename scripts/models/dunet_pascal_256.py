'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, AveragePooling2D,ZeroPadding2D, Reshape, Concatenate, concatenate, BatchNormalization, Dropout, UpSampling2D, Add
from keras.regularizers import l2
import keras.backend as K


from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

def dronenet(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False,
            bottleneck=True,
            reduction=0.0,
            dropout_rate=None,
            weight_decay=1e-4):


    n_predictor_layers = 4 # The number of predictor conv layers in the network is 6 for the original SSD300.
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels), name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels), name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(x1)




    def dense_block(prevDense, stage, nb_layers, nb_filter, growth_rate, bottleneck=True, dropout_rate=None, weight_decay=1e-4,
                    grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''

        for i in range(nb_layers):
            branch = i + 1
            dense = conv_block(prevDense, stage, branch, growth_rate, bottleneck, dropout_rate, weight_decay)
            #print('layer', stage, branch, nb_filter, prevDense.shape)
            prevDense = concatenate([prevDense, dense], axis=3, name='concat_'+str(stage)+'_'+str(branch))
            #print('concate', stage, nb_filter, prevDense.shape)

            if grow_nb_filters:
                nb_filter += growth_rate
        #print('dense', stage, nb_filter, prevDense.shape)
        return prevDense, nb_filter




    def conv_block(prevConv, stage, branch, nb_filter, bottleneck=True, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                prevConv: input tensor
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        prevConv = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_x1_bn')(prevConv)
        prevConv = Activation('relu', name=relu_name_base+'_x1')(prevConv)

        if bottleneck:
            inter_channel = nb_filter * 4  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
            prevConv = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal',
                          padding='same', use_bias=False, kernel_regularizer=l2(weight_decay), name=conv_name_base+'_x1')(prevConv)

            if dropout_rate:
                prevConv = Dropout(dropout_rate)(prevConv)

        prevConv = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_x2_bn')(prevConv)
        prevConv = Activation('relu', name=relu_name_base+'_x2')(prevConv)
        prevConv = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False, name=conv_name_base+'_x2')(prevConv)

        if dropout_rate:
            prevConv = Dropout(dropout_rate)(prevConv)

        return prevConv



    def transition_block(prevTran, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
            # Arguments
                prevTran: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''

        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'poolD' + str(stage)

        prevTran = BatchNormalization(epsilon=eps, axis=3, name=conv_name_base+'_bn')(prevTran)
        prevTran = Activation('relu', name=relu_name_base)(prevTran)
        prevTran = Conv2D(int(nb_filter * compression), (1, 1), activation='relu', kernel_initializer='he_normal',
                          padding='same', use_bias=False, kernel_regularizer=l2(weight_decay), name=conv_name_base)(prevTran)

        if dropout_rate:
            prevTran = Dropout(dropout_rate)(prevTran)

        # if stage !=3:
        #     prevTran = MaxPooling2D(pool_size=(2, 2), name=pool_name_base)(prevTran)
        #print('tran', stage, prevTran.shape)
        return prevTran



    # DenseNet Parameters

    eps = 1.1e-5
    nb_filter =64
    t_nb_filter = 256
    growth_rate = 32
    nb_layers = [5, 7, 7, 7]
    compression = 1.0 - reduction
    conv1_1 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    conv1_2 = BatchNormalization(epsilon=eps, axis=3, name='conv1_2_bn')(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_2)
    conv1_3 = BatchNormalization(epsilon=eps, axis=3, name='conv1_3_bn')(conv1_2)
    conv1_3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_3')(conv1_3)
    pool1_3 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1_3')(conv1_3)

    # Add densBlock1
    stage = 1
    conv1, nb_filter = dense_block(pool1_3, stage, nb_layers[0], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=None, weight_decay=weight_decay)
    trans1 = transition_block(conv1, stage, t_nb_filter, compression=compression, weight_decay=weight_decay)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(trans1)
    nb_filter = int(nb_filter * compression)


    # Add densBlock2
    stage = 2
    conv2, nb_filter = dense_block(pool1, stage, nb_layers[1], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=None, weight_decay=weight_decay)
    trans2 = transition_block(conv2, stage, t_nb_filter, compression=compression, weight_decay=weight_decay)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(trans2)
    nb_filter = int(nb_filter * compression)


    # Add densBlock3
    stage = 3
    conv3, nb_filter = dense_block(pool2, stage, nb_layers[2], nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=None, weight_decay=weight_decay)
    trans3 = transition_block(conv3, stage, t_nb_filter, compression=compression, weight_decay=weight_decay)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(trans3)
    nb_filter = int(nb_filter * compression)


    # Add densBlock4
    stage = 4
    conv4, nb_filter = dense_block(pool3, stage, nb_layers[3], nb_filter, 26, bottleneck=bottleneck,
                                   dropout_rate=None, weight_decay=weight_decay)
    trans4 = transition_block(conv4, stage, t_nb_filter, compression=compression, weight_decay=weight_decay)
    #pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(trans4)




    M5 = BatchNormalization(epsilon=eps, axis=3, name='m5_bn1')(trans4)

    M5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='M5P')(M5)

    M4 = UpSampling2D(size=(2, 2))(M5)

    # M4E = ZeroPadding2D(padding=((0, 1), (1, 0)), name='trans3_padding')(trans3)

    M4 = Add()([M4, trans3])

    M4 = BatchNormalization(epsilon=eps, axis=3, name='M4_bn')(M4)

    M4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='M4P')(M4)

    M3 = UpSampling2D(size=(2, 2))(M4)

    # M3E = ZeroPadding2D(padding=((1, 1), (1, 1)), name='trans3_padding')(trans2)

    M3 = Add()([M3, trans2])

    M3 = BatchNormalization(epsilon=eps, axis=3, name='M3_bn')(M3)

    M3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='M3P')(M3)

    M2 = UpSampling2D(size=(2, 2))(M3)

    # M2E = ZeroPadding2D(padding=((2, 2), (2, 2)), name='trans3_padding')(trans1)

    M2 = Add()([M2, trans1])

    M2 = BatchNormalization(epsilon=eps, axis=3, name='M2_bn')(M2)

    M2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='M2P')(M2)

    # print('M5', M5.shape)
    # print('M4', M4.shape)
    # print('M3', M3.shape)
    # print('M2', M2.shape)
    # print()


    ### Build the convolutional predictor layers on top of the base network
    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    # conv6_2_mbox = BatchNormalization(epsilon=eps, axis=3, name='conv6_2_mbox_conf_bn1')(M2)
    # conv6_2_mbox = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf1')(conv6_2_mbox)
    conv6_2_mbox_conf = BatchNormalization(epsilon=eps, axis=3, name='conv6_2_mbox_conf_bn2')(M2)
    conv6_2_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf2')(conv6_2_mbox_conf)

    # conv7_2_mbox = BatchNormalization(epsilon=eps, axis=3, name='conv7_2_mbox_conf_bn1')(M3)
    # conv7_2_mbox = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf1')(conv7_2_mbox)
    conv7_2_mbox_conf = BatchNormalization(epsilon=eps, axis=3, name='conv7_2_mbox_conf_bn2')(M3)
    conv7_2_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf2')(conv7_2_mbox_conf)

    # conv8_2_mbox = BatchNormalization(epsilon=eps, axis=3, name='conv8_2_mbox_conf_bn1')(M4)
    # conv8_2_mbox = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf1')(conv8_2_mbox)
    conv8_2_mbox_conf = BatchNormalization(epsilon=eps, axis=3, name='conv8_2_mbox_conf_bn2')(M4)
    conv8_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf2')(conv8_2_mbox_conf)

    # conv9_2_mbox = BatchNormalization(epsilon=eps, axis=3, name='conv9_2_mbox_conf_bn1')(M5)
    # conv9_2_mbox = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf1')(conv9_2_mbox)
    conv9_2_mbox_conf = BatchNormalization(epsilon=eps, axis=3, name='conv9_2_mbox_conf_bn2')(M5)
    conv9_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf2')(conv9_2_mbox_conf)


    # print('conv6_2_mbox_conf', conv6_2_mbox_conf.shape)
    # print('conv7_2_mbox_conf', conv7_2_mbox_conf.shape)
    # print('conv8_2_mbox_conf', conv8_2_mbox_conf.shape)
    # print('conv9_2_mbox_conf', conv9_2_mbox_conf.shape)
    # print()

    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    conv6_2_mbox_loc = BatchNormalization(epsilon=eps, axis=3, name='conv6_2_mbox_loc_bn')(M2)
    conv6_2_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2_mbox_loc)
    conv7_2_mbox_loc = BatchNormalization(epsilon=eps, axis=3, name='conv7_2_mbox_loc_bn')(M3)
    conv7_2_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2_mbox_loc)
    conv8_2_mbox_loc = BatchNormalization(epsilon=eps, axis=3, name='conv8_2_mbox_loc_bn')(M4)
    conv8_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2_mbox_loc)
    conv9_2_mbox_loc = BatchNormalization(epsilon=eps, axis=3, name='conv9_2_mbox_loc_bn')(M5)
    conv9_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2_mbox_loc)


    # print('conv6_2_mbox_loc', conv6_2_mbox_loc.shape)
    # print('conv7_2_mbox_loc', conv7_2_mbox_loc.shape)
    # print('conv8_2_mbox_loc', conv8_2_mbox_loc.shape)
    # print('conv9_2_mbox_loc', conv9_2_mbox_loc.shape)
    # print()

    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)


    # print('conv6_2_mbox_priorbox', conv6_2_mbox_priorbox.shape)
    # print('conv7_2_mbox_priorbox', conv7_2_mbox_priorbox.shape)
    # print('conv8_2_mbox_priorbox', conv8_2_mbox_priorbox.shape)
    # print('conv9_2_mbox_priorbox', conv9_2_mbox_priorbox.shape)
    # print()

    ### Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them

    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)


    # print('conv6_2_mbox_conf_reshape', conv6_2_mbox_conf_reshape.shape)
    # print('conv7_2_mbox_conf_reshape', conv7_2_mbox_conf_reshape.shape)
    # print('conv8_2_mbox_conf_reshape', conv8_2_mbox_conf_reshape.shape)
    # print('conv9_2_mbox_conf_reshape', conv9_2_mbox_conf_reshape.shape)
    # print()

    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss

    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)


    # print('conv6_2_mbox_loc_reshape', conv6_2_mbox_loc_reshape.shape)
    # print('conv7_2_mbox_loc_reshape', conv7_2_mbox_loc_reshape.shape)
    # print('conv8_2_mbox_loc_reshape', conv8_2_mbox_loc_reshape.shape)
    # print('conv9_2_mbox_loc_reshape', conv9_2_mbox_loc_reshape.shape)
    # print()

    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`

    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)


    # print('conv6_2_mbox_priorbox_reshape', conv6_2_mbox_priorbox_reshape.shape)
    # print('conv7_2_mbox_priorbox_reshape', conv7_2_mbox_priorbox_reshape.shape)
    # print('conv8_2_mbox_priorbox_reshape', conv8_2_mbox_priorbox_reshape.shape)
    # print('conv9_2_mbox_priorbox_reshape', conv9_2_mbox_priorbox_reshape.shape)
    # print()

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape
                                                       ,conv9_2_mbox_conf_reshape
                                                       ])

    # print('mbox_conf', mbox_conf.shape)
    # print()

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape
                                                     ,conv9_2_mbox_loc_reshape
                                                     ])

    # print('mbox_loc', mbox_loc.shape)
    # print()

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape
                                                               ,conv9_2_mbox_priorbox_reshape
                                                               ])

    # print('mbox_priorbox', mbox_priorbox.shape)
    # print()

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # print('mbox_conf_softmax', mbox_conf_softmax.shape)
    # print()

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    # print('predictions', predictions.shape)

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError("`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([ conv6_2_mbox_conf._keras_shape[1:3],
                                     conv7_2_mbox_conf._keras_shape[1:3],
                                     conv8_2_mbox_conf._keras_shape[1:3],
                                     conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
