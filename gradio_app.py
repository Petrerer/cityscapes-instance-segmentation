import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model(
    "saved_models/final_model/model3_best.h5",
    compile=False
)

def predict(image):
    img = cv2.resize(image, (512, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)


    outputs = model.predict(img)
    pred = outputs[0]

    mask = np.argmax(pred, axis=-1)
    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    class_colors = {
        1: [255, 0, 0],
        2: [0, 255, 0]
    }

    overlay = image.copy()

    for cls, color in class_colors.items():
        overlay[mask == cls] = color

    return overlay


gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Semantic Segmentation Demo"
).launch()
