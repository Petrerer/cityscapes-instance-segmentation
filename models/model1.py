import tensorflow as tf
from tensorflow.keras import layers, models

IMG_HEIGHT = 256
IMG_WIDTH = 512

# A standard U-Net architecture with attention mechanism

def build_model1():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(256, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(512, 3, padding='same', activation='relu')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    
    bottleneck = layers.Conv2D(1024, 3, padding='same', activation='relu')(p4)
    bottleneck = layers.Conv2D(1024, 3, padding='same', activation='relu')(bottleneck)
    
    u1 = layers.UpSampling2D(2)(bottleneck)
    u1 = layers.Concatenate()([u1, c4])
    u1 = layers.Conv2D(512, 3, padding='same', activation='relu')(u1)
    u1 = layers.Conv2D(512, 3, padding='same', activation='relu')(u1)
    
    u2 = layers.UpSampling2D(2)(u1)
    u2 = layers.Concatenate()([u2, c3])
    u2 = layers.Conv2D(256, 3, padding='same', activation='relu')(u2)
    u2 = layers.Conv2D(256, 3, padding='same', activation='relu')(u2)
    
    u3 = layers.UpSampling2D(2)(u2)
    u3 = layers.Concatenate()([u3, c2])
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    u3 = layers.Conv2D(128, 3, padding='same', activation='relu')(u3)
    
    u4 = layers.UpSampling2D(2)(u3)
    u4 = layers.Concatenate()([u4, c1])
    u4 = layers.Conv2D(64, 3, padding='same', activation='relu')(u4)
    u4 = layers.Conv2D(64, 3, padding='same', activation='relu')(u4)
    
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(u4)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(u4)
    attention = layers.Concatenate()([avg_pool, max_pool])
    attention = layers.Conv2D(64, 1, activation='sigmoid')(attention)
    u4 = layers.Multiply()([u4, attention])
    
    semantic_output = layers.Conv2D(3, 1, activation='softmax', name='semantic')(u4)
    
    model = models.Model(inputs=inputs, outputs=semantic_output)
    return model