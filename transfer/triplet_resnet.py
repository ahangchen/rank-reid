import utils.cuda_util
import numpy as np
import keras.backend as K
from keras import layers, Model, Input
from keras.layers import BatchNormalization, Conv2D, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.models import load_model


def load_split_resnet(resnet_path):
    resnet = load_model(resnet_path).layers[2]
    for layer in resnet.layers:
        if isinstance(layer, BatchNormalization):
            a = layer.get_weights()
            b = layer.get_weights()
            layer.set_weights(layer.get_weights())
        elif isinstance(layer, Conv2D):
            avg_w = list()
            w1 = layer.get_weights()
            w2 = layer.get_weights()
            layer.set_weights(layer.get_weights())


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    input_tensor0, input_tensor1, input_tensor2 = input_tensor

    conv_a = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')

    x0 = conv_a(input_tensor0)
    x1 = conv_a(input_tensor1)
    x2 = conv_a(input_tensor2)
    x0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_2')(x2)

    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)

    conv_b = Conv2D(filters2, kernel_size,
                    padding='same', name=conv_name_base + '2b')
    x0 = conv_b(x0)
    x1 = conv_b(x1)
    x2 = conv_b(x2)
    x0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_2')(x2)
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)

    conv_c = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')
    x0 = conv_c(x0)
    x1 = conv_c(x1)
    x2 = conv_c(x2)
    x0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_2')(x2)

    x0 = layers.add([x0, input_tensor0])
    x1 = layers.add([x1, input_tensor1])
    x2 = layers.add([x2, input_tensor2])
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    return x0, x1, x2


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    input_tensor0, input_tensor1, input_tensor2 = input_tensor
    conv_a = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')
    x0 = conv_a(input_tensor0)
    x1 = conv_a(input_tensor1)
    x2 = conv_a(input_tensor2)
    x0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_2')(x2)
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)

    conv_b = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')
    x0 = conv_b(x0)
    x1 = conv_b(x1)
    x2 = conv_b(x2)
    x0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_2')(x2)
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)

    conv_c = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')
    x0 = conv_c(x0)
    x1 = conv_c(x1)
    x2 = conv_c(x2)
    x0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_2')(x2)

    conv_shot = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')
    shortcut0 = conv_shot(input_tensor0)
    shortcut1 = conv_shot(input_tensor1)
    shortcut2 = conv_shot(input_tensor2)
    shortcut0 = BatchNormalization(axis=bn_axis, name=bn_name_base + '1_0')(shortcut0)
    shortcut1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '1_1')(shortcut1)
    shortcut2 = BatchNormalization(axis=bn_axis, name=bn_name_base + '1_2')(shortcut2)

    x0 = layers.add([x0, shortcut0])
    x1 = layers.add([x1, shortcut1])
    x2 = layers.add([x2, shortcut2])
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    return x0, x1, x2


def Triplet_ResNet50(resnet):

    print(resnet.summary())
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img0 = Input(shape=(224, 224, 3), name='img_0')
    img1 = Input(shape=(224, 224, 3), name='img_1')
    img2 = Input(shape=(224, 224, 3), name='img_2')

    x0 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_0')(img0)
    x1 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_1')(img1)
    x2 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_2')(img2)

    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')
    x0 = conv1(x0)
    x1 = conv1(x1)
    x2 = conv1(x2)

    x0 = BatchNormalization(axis=bn_axis, name='bn_conv1_0')(x0)
    x1 = BatchNormalization(axis=bn_axis, name='bn_conv1_1')(x1)
    x2 = BatchNormalization(axis=bn_axis, name='bn_conv1_2')(x2)

    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x2 = Activation('relu')(x2)
    x0 = MaxPooling2D((3, 3), strides=(2, 2))(x0)
    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x1)
    x2 = MaxPooling2D((3, 3), strides=(2, 2))(x2)

    x0, x1, x2 = conv_block([x0, x1, x2], 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [64, 64, 256], stage=2, block='b')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [64, 64, 256], stage=2, block='c')

    x0, x1, x2 = conv_block([x0, x1, x2], 3, [128, 128, 512], stage=3, block='a')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [128, 128, 512], stage=3, block='b')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [128, 128, 512], stage=3, block='c')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [128, 128, 512], stage=3, block='d')

    x0, x1, x2 = conv_block([x0, x1, x2], 3, [256, 256, 1024], stage=4, block='a')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [256, 256, 1024], stage=4, block='b')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [256, 256, 1024], stage=4, block='c')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [256, 256, 1024], stage=4, block='d')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [256, 256, 1024], stage=4, block='e')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [256, 256, 1024], stage=4, block='f')

    x0, x1, x2 = conv_block([x0, x1, x2], 3, [512, 512, 2048], stage=5, block='a')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [512, 512, 2048], stage=5, block='b')
    x0, x1, x2 = identity_block([x0, x1, x2], 3, [512, 512, 2048], stage=5, block='c')

    last_avg_pool = AveragePooling2D((7, 7), name='avg_pool')

    x0 = last_avg_pool(x0)
    x1 = last_avg_pool(x1)
    x2 = last_avg_pool(x2)

    # Create model.
    model = Model(inputs=[img0, img1, img2], outputs=[x0, x1, x2], name='resnet50')

    for layer in resnet.layers:
        if isinstance(layer, Conv2D):
            print 'set Conv2D weights'
            model.get_layer(name=layer.name).set_weights(layer.get_weights())
        elif isinstance(layer, BatchNormalization):
            print 'set BN weights'
            model.get_layer(name=layer.name+'_0').set_weights(layer.get_weights())
            model.get_layer(name=layer.name+'_1').set_weights(layer.get_weights())
            model.get_layer(name=layer.name+'_2').set_weights(layer.get_weights())
        elif len(layer.get_weights()) > 0:
            print layer.name
    print model.summary()
    return model


if __name__ == '__main__':
    load_split_resnet('../pretrain/cuhk_pair_pretrain.h5')
