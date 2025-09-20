import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


model = tf.keras.models.load_model("butterfly_model.h5")


class_names = 


def preprocess_and_predict(img):
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return {class_names[predicted_class]: float(confidence)}


iface = gr.Interface(
    fn=preprocess_and_predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🦋 Butterfly Classification",
    description="Upload a butterfly image and the model will predict its species."
)

iface.launch()
