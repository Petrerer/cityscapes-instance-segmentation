import tensorflow as tf
from tensorflow.keras import layers, models

IMG_HEIGHT = 256
IMG_WIDTH = 512

# A model that uses dilated convolutions in a deep semantic stack

def build_model3():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    d1 = layers.Conv2D(256, 3, dilation_rate=2, padding='same', activation='relu')(p2)
    d1 = layers.Conv2D(256, 3, dilation_rate=2, padding='same', activation='relu')(d1)

    d2 = layers.Conv2D(256, 3, dilation_rate=4, padding='same', activation='relu')(d1)
    d2 = layers.Conv2D(256, 3, dilation_rate=4, padding='same', activation='relu')(d2)

    d3 = layers.Conv2D(256, 3, dilation_rate=8, padding='same', activation='relu')(d2)

    u1 = layers.UpSampling2D(2)(d3)
    u1 = layers.Concatenate()([u1, c2])
    u1 = layers.Conv2D(128, 3, padding='same', activation='relu')(u1)
    u1 = layers.Conv2D(128, 3, padding='same', activation='relu')(u1)

    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Concatenate()([u2, c1])
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)
    u2 = layers.Conv2D(64, 3, padding='same', activation='relu')(u2)

    outputs = layers.Conv2D(3, 1, activation='softmax')(u2)
    return models.Model(inputs, outputs)
