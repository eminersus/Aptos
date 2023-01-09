import tensorflow as tf
from keras import layers
from keras.utils import image_dataset_from_directory
from keras import Sequential, callbacks, regularizers

class Data_set:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ds_train = None
        self.ds_validation = None

    def get_data(self):
        self.ds_train = image_dataset_from_directory(
            self.data_dir + '//train',
            labels='inferred',
            shuffle=True,
            label_mode='categorical',
            batch_size=64,
            interpolation='nearest',
            class_names=['happy', 'neutral', 'sad', 'surprise'],
            color_mode='grayscale',
            image_size=(48, 48)
        )

        self.ds_validation = image_dataset_from_directory(
            self.data_dir + '//validation',
            labels='inferred',
            shuffle=True,
            label_mode='categorical',
            batch_size=64,
            interpolation='nearest',
            color_mode='grayscale',
            image_size=(48, 48)
        )