import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from keras import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.engine import Model
from keras.layers import Lambda, Dense, Dropout, Flatten
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model, to_categorical
from keras.applications.resnet50 import preprocess_input
from numpy.random import randint, shuffle, choice


def reid_data_prepare(data_list_path, train_dir_path):
    class_img_labels = dict()
    cur_label = -2
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img, lbl = line.split()
            img = image.load_img(os.path.join(train_dir_path, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            if int(lbl) != cur_label:
                cur_list = list()
                class_img_labels[lbl] = cur_list
                cur_label = int(lbl)
            class_img_labels[lbl].append(img[0])
    return class_img_labels


def grid_data_prepare(data_list_path, train_dir_path):
    class_img_labels = dict()
    cur_label = -2
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            img = image.load_img(os.path.join(train_dir_path, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            if int(lbl) != cur_label:
                cur_list = list()
                class_img_labels[lbl] = cur_list
                cur_label = int(lbl)
            class_img_labels[lbl].append(img[0])
    return class_img_labels


def pair_generator(class_img_labels, batch_size, train=False):
    cur_epoch = 0
    pos_prop = 4
    while True:
        left_label = randint(len(class_img_labels), size=batch_size)
        if cur_epoch % pos_prop == 0:
            right_label = left_label
        else:
            right_label = np.copy(left_label)
            shuffle(right_label)
        # select by label
        left_images = list()
        right_images = list()
        if train:
            slice_start = 0
        else:
            # val
            slice_start = 0.9
        for i in range(batch_size):
            len_left_label_i = len(class_img_labels[str(left_label[i])])
            left_images.append(class_img_labels[str(left_label[i])][int(slice_start * len_left_label_i):][
                                   choice(len_left_label_i - int(len_left_label_i * slice_start))])
            len_right_label_i = len(class_img_labels[str(right_label[i])])
            right_images.append(class_img_labels[str(right_label[i])][int(slice_start * len_right_label_i):][
                                    choice(len_right_label_i - int(len_right_label_i * slice_start))])

        left_images = np.array(left_images)
        right_images = np.array(right_images)
        binary_label = to_categorical((left_label == right_label).astype(int), num_classes=2)
        left_label = to_categorical(left_label, num_classes=len(class_img_labels))
        right_label = to_categorical(right_label, num_classes=len(class_img_labels))
        cur_epoch += 1
        yield [left_images, right_images], [left_label, right_label, binary_label]


def eucl_dist(inputs):
    x, y = inputs
    return (x - y) ** 2


def pair_model(single_model_path, num_classes):
    base_model = load_model(single_model_path)
    base_model = Model(inputs=base_model.input, outputs=[base_model.get_layer('avg_pool').output], name='resnet50')
    img1 = Input(shape=(224, 224, 3), name='img_1')
    img2 = Input(shape=(224, 224, 3), name='img_2')
    feature1 = Flatten()(base_model(img1))
    feature2 = Flatten()(base_model(img2))
    dis = Lambda(eucl_dist, name='square')([feature1, feature2])
    judge = Dense(2, activation='softmax', name='bin_out')(dis)
    category_predict1 = Dense(num_classes, activation='softmax', name='ctg_out_1')(
        Dropout(0.5)(feature1)
    )
    category_predict2 = Dense(num_classes, activation='softmax', name='ctg_out_2')(
        Dropout(0.5)(feature2)
    )
    model = Model(inputs=[img1, img2], outputs=[category_predict1, category_predict2, judge])
    plot_model(model, to_file='model_combined.png')
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    return model


def pair_tune(train_generator, val_generator, source_model_path, batch_size=48, num_classes=751):
    model = pair_model(source_model_path, num_classes)
    model.compile(optimizer='nadam',
                  loss={'ctg_out_1': 'categorical_crossentropy',
                        'ctg_out_2': 'categorical_crossentropy',
                        'bin_out': 'categorical_crossentropy'},
                  loss_weights={
                      'ctg_out_1': 1.,
                      'ctg_out_2': 1.,
                      'bin_out': 1.
                  },
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.0001,
                                cooldown=0, min_lr=0)
    save_model = ModelCheckpoint('resnet50-{epoch:02d}-{val_ctg_out_1_acc:.2f}.h5', period=2)
    model.fit_generator(train_generator,
                        steps_per_epoch=16500 / batch_size + 1,
                        epochs=30,
                        validation_data=val_generator,
                        validation_steps=1800 / batch_size + 1,
                        callbacks=[early_stopping, auto_lr, save_model])
    model.save('pair_pretrain.h5')


if __name__ == '__main__':
    DATASET = '../dataset/Market'
    LIST = os.path.join(DATASET, 'pretrain.list')
    TRAIN = os.path.join(DATASET, 'bounding_box_train')
    class_count = 751
    class_img_labels = reid_data_prepare(LIST, TRAIN)
    batch_size = 64
    pair_tune(
        pair_generator(class_img_labels, batch_size=batch_size, train=True),
        pair_generator(class_img_labels, batch_size=batch_size, train=False),
        '../baseline/0.ckpt',
        batch_size=batch_size, num_classes=class_count
    )
