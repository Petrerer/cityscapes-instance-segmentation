# models/model2.py
import tensorflow as tf
from tensorflow.keras import layers, models

#A model thats shallower but wider than model1

IMG_HEIGHT = 256
IMG_WIDTH = 512
def build_model2():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Encoder (shallower but wide)
    c1 = layers.Conv2D(96, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(96, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(192, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(192, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    # Bottleneck (wide to compensate for lost depth)
    b = layers.Conv2D(384, 3, padding='same', activation='relu')(p2)
    b = layers.Conv2D(384, 3, padding='same', activation='relu')(b)

    # Decoder
    u1 = layers.UpSampling2D(2)(b)
    u1 = layers.Concatenate()([u1, c2])
    u1 = layers.Conv2D(192, 3, padding='same', activation='relu')(u1)
    u1 = layers.Conv2D(192, 3, padding='same', activation='relu')(u1)

    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Concatenate()([u2, c1])
    u2 = layers.Conv2D(96, 3, padding='same', activation='relu')(u2)
    u2 = layers.Conv2D(96, 3, padding='same', activation='relu')(u2)

    # Attention
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(u2)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(u2)
    attn = layers.Concatenate()([avg_pool, max_pool])
    attn = layers.Conv2D(96, 1, activation='sigmoid')(attn)
    u2 = layers.Multiply()([u2, attn])

    outputs = layers.Conv2D(3, 1, activation='softmax')(u2)
    return models.Model(inputs, outputs)
