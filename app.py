import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Detector COVID", layout="centered")

st.title("🩺 Detector de Enfermedades Pulmonares")
st.write("Sube una radiografía de tórax")

model = tf.keras.models.load_model("modelo_covid.keras")

class_names = ['COVID', 'LUNG_OPACITY', 'NORMAL', 'PNEUMONIA']

uploaded_file = st.file_uploader("📤 Subir imagen", type=["jpg","png","jpeg"])

if uploaded_file is not None:

```
img = image.load_img(uploaded_file, target_size=(224, 224), color_mode='rgb')
st.image(img, caption="Imagen subida", use_container_width=True)

img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

resultado = class_names[np.argmax(score)]
confianza = round(100*np.max(score),2)

st.write("## Resultado:")

if resultado == "NORMAL":
    st.success(f"🟢 {resultado} ({confianza}%)")
elif resultado == "COVID":
    st.error(f"🔴 {resultado} ({confianza}%)")
else:
    st.warning(f"🟡 {resultado} ({confianza}%)")

st.progress(int(confianza))
