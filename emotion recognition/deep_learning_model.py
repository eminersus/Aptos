import tensorflow as tf
from keras import layers
from keras.utils import image_dataset_from_directory
from keras import Sequential, callbacks, regularizers
import data_set

class Model:

    def __init__(self):
        self.model = None
        self.callbacks = None
    
    def create_model(self):
        self.model = Sequential([
            layers.InputLayer(input_shape=(48, 48, 1)),
            layers.Rescaling(scale=1./255),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.4),
            layers.RandomFlip(mode = "horizontal"),

            layers.Conv2D(filter=32, padding='same', kernel_size=(3, 3), activation='relu'),
            layers.Conv2D(filter=64, padding='same', kernel_size=(3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(filter=128, activation='relu', kernel_size= (3,3), padding ='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(filter=512, activation='relu', kernel_size= (3,3), padding ='same', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Conv2D(filter=512, activation='relu', kernel_size= (3,3), padding ='same', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),

            layers.Flatten(),
            layers.Dense(256,activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(4, activation='softmax')
        ])

    def compile(self):
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = tf.keras.optimizers.Adam(epsilon=0.0001, decay = 1e-6),
            metrics = ['accuracy']
        )

    def set_callbacks(self):
        reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.2, verbose=1, min_delta=0.0001)
        checkpoint = callbacks.ModelCheckpoint(filepath = "saved_model", verbose = 1,save_weights_only=True, save_best_only = True, mode = "min", monitor = "val_loss")
        self.callbacks = [reduce, checkpoint]

    def history(self, ds_train, ds_validation):
        self.model.fit(
            ds_train,
            validation_data=ds_validation,
            epochs=90,
            callbacks=self.callbacks
        )

    def save_model(self):
        model_json = self.model.to_json()
        with open("emotion_model.json", "w") as json_file:
            json_file.write(model_json)