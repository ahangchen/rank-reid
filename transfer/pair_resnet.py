import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import keras.backend as K
from keras import layers, Model, Input
from keras.layers import BatchNormalization, Conv2D, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.models import load_model




def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    input_tensor0, input_tensor1 = input_tensor

    conv_a = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')

    x0 = conv_a(input_tensor0)
    x1 = conv_a(input_tensor1)

    bn_2a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
    x0 = bn_2a(x0)
    x1 = bn_2a(x1)

    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)

    conv_b = Conv2D(filters2, kernel_size,
                    padding='same', name=conv_name_base + '2b')
    x0 = conv_b(x0)
    x1 = conv_b(x1)

    bn_2b = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
    x0 = bn_2b(x0)
    x1 = bn_2b(x1)
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)

    conv_c = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')
    x0 = conv_c(x0)
    x1 = conv_c(x1)

    bn_2c = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
    x0 = bn_2c(x0)
    x1 = bn_2c(x1)

    x0 = layers.add([x0, input_tensor0])
    x1 = layers.add([x1, input_tensor1])
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    return x0, x1


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    input_tensor0, input_tensor1 = input_tensor
    conv_a = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')
    x0 = conv_a(input_tensor0)
    x1 = conv_a(input_tensor1)

    bn_2a = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')
    x0 = bn_2a(x0)
    x1 = bn_2a(x1)
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)

    conv_b = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')
    x0 = conv_b(x0)
    x1 = conv_b(x1)

    bn_2b = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')
    x0 = bn_2b(x0)
    x1 = bn_2b(x1)
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)

    conv_c = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')
    x0 = conv_c(x0)
    x1 = conv_c(x1)
    bn_2c = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')
    x0 = bn_2c(x0)
    x1 = bn_2c(x1)

    conv_shot = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')
    shortcut0 = conv_shot(input_tensor0)
    shortcut1 = conv_shot(input_tensor1)
    bn_1 = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')
    shortcut0 = bn_1(shortcut0)
    shortcut1 = bn_1(shortcut1)

    x0 = layers.add([x0, shortcut0])
    x1 = layers.add([x1, shortcut1])
    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    return x0, x1


def Pair_ResNet50(resnet):

    print(resnet.summary())
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img0 = Input(shape=(224, 224, 3), name='img_0')
    img1 = Input(shape=(224, 224, 3), name='img_1')

    x0 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_0')(img0)
    x1 = ZeroPadding2D(padding=(3, 3), name='conv1_pad_1')(img1)

    conv1 = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1')
    x0 = conv1(x0)
    x1 = conv1(x1)

    bn_conv1 = BatchNormalization(axis=bn_axis, name='bn_conv1')
    x0 = bn_conv1(x0)
    x1 = bn_conv1(x1)

    x0 = Activation('relu')(x0)
    x1 = Activation('relu')(x1)
    x0 = MaxPooling2D((3, 3), strides=(2, 2))(x0)
    x1 = MaxPooling2D((3, 3), strides=(2, 2))(x1)

    x0, x1 = conv_block([x0, x1], 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x0, x1 = identity_block([x0, x1], 3, [64, 64, 256], stage=2, block='b')
    x0, x1 = identity_block([x0, x1], 3, [64, 64, 256], stage=2, block='c')

    x0, x1 = conv_block([x0, x1], 3, [128, 128, 512], stage=3, block='a')
    x0, x1 = identity_block([x0, x1], 3, [128, 128, 512], stage=3, block='b')
    x0, x1 = identity_block([x0, x1], 3, [128, 128, 512], stage=3, block='c')
    x0, x1 = identity_block([x0, x1], 3, [128, 128, 512], stage=3, block='d')

    x0, x1 = conv_block([x0, x1], 3, [256, 256, 1024], stage=4, block='a')
    x0, x1 = identity_block([x0, x1], 3, [256, 256, 1024], stage=4, block='b')
    x0, x1 = identity_block([x0, x1], 3, [256, 256, 1024], stage=4, block='c')
    x0, x1 = identity_block([x0, x1], 3, [256, 256, 1024], stage=4, block='d')
    x0, x1 = identity_block([x0, x1], 3, [256, 256, 1024], stage=4, block='e')
    x0, x1 = identity_block([x0, x1], 3, [256, 256, 1024], stage=4, block='f')

    x0, x1 = conv_block([x0, x1], 3, [512, 512, 2048], stage=5, block='a')
    x0, x1 = identity_block([x0, x1], 3, [512, 512, 2048], stage=5, block='b')
    x0, x1 = identity_block([x0, x1], 3, [512, 512, 2048], stage=5, block='c')

    last_avg_pool = AveragePooling2D((7, 7), name='avg_pool')

    x0 = last_avg_pool(x0)
    x1 = last_avg_pool(x1)

    # Create model.
    model = Model(inputs=[img0, img1], outputs=[x0, x1], name='resnet50')

    for layer in resnet.layers:
        if isinstance(layer, Conv2D):
            print 'set Conv2D weights'
            model.get_layer(name=layer.name).set_weights(layer.get_weights())
        elif isinstance(layer, BatchNormalization):
            print 'set BN weights'
            model.get_layer(name=layer.name).set_weights(layer.get_weights())
            model.get_layer(name=layer.name).set_weights(layer.get_weights())
            model.get_layer(name=layer.name).set_weights(layer.get_weights())
        elif len(layer.get_weights()) > 0:
            print layer.name
    print model.summary()
    return model


if __name__ == '__main__':
    Pair_ResNet50(load_model('../pretrain/cuhk_pair_pretrain.h5').layers[2])
