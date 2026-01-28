import os
import numpy as np
import tensorflow as tf
import random

IMG_HEIGHT = 256
IMG_WIDTH = 512

def build_dataset(images_dir, masks_dir, batch_size, shuffle=False, augment=False, capped_size=-1):
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.npy'))])
    
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    mask_paths = [os.path.join(masks_dir, f) for f in mask_files]

    if capped_size!=-1:
        image_paths = image_paths[:capped_size]
        mask_paths = mask_paths[:capped_size]

    def load_data(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])

        mask = tf.numpy_function(np.load, [mask_path], tf.int32)
        mask.set_shape([IMG_HEIGHT, IMG_WIDTH])

        semantic = tf.zeros([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32)
        semantic = tf.where(mask >= 2000, 2, semantic)  # class 2
        semantic = tf.where((mask >= 1000) & (mask < 2000), 1, semantic)  # class 1

        return image, semantic

    def augment_data(image, semantic):
        if tf.random.uniform(()) > 0.3:
            # Horizontal flip
            image = tf.image.flip_left_right(image)
            semantic = tf.expand_dims(semantic, -1)
            semantic = tf.image.flip_left_right(semantic)
            semantic = tf.squeeze(semantic, -1)

        # Random brightness/contrast (images only)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Clip to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, semantic

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        ds = ds.shuffle(1000)
    
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
