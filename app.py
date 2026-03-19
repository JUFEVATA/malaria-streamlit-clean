import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Clasificador de Malaria",
    page_icon="🦠",
    layout="centered"
)

CLASS_NAMES = ["Parasitized", "Uninfected"]
IMG_SIZE = (224, 224)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("artifacts/lenet.keras")

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image: Image.Image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)

    if prediction.shape[-1] == 1:
        score = float(prediction[0][0])
        label = CLASS_NAMES[0] if score >= 0.5 else CLASS_NAMES[1]
        confidence = score if score >= 0.5 else 1 - score
    else:
        pred_idx = int(np.argmax(prediction, axis=1)[0])
        label = CLASS_NAMES[pred_idx]
        confidence = float(np.max(prediction))

    return label, confidence * 100, prediction.shape

st.title("🦠 Clasificador de Malaria")
st.markdown(
    """
    Esta aplicación permite cargar una imagen de una célula sanguínea y clasificarla
    con un modelo CNN tipo LeNet.

    **Clases:**
    - **Parasitized**
    - **Uninfected**
    """
)

try:
    model = load_model()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Sube una imagen de célula (JPG o PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    try:
        img = Image.open(uploaded)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Imagen cargada", use_container_width=True)

        with col2:
            st.write("### Información")
            st.write(f"**Nombre:** {uploaded.name}")
            st.write(f"**Tipo:** {uploaded.type}")
            st.write(f"**Tamaño:** {round(uploaded.size / 1024, 2)} KB")

        if st.button("Predecir", use_container_width=True):
            with st.spinner("Procesando imagen..."):
                label, confidence, prediction_shape = predict_image(model, img)

            st.write("## Resultado")

            if label == "Parasitized":
                st.error(f"**Clase predicha:** {label}")
            else:
                st.success(f"**Clase predicha:** {label}")

            st.metric("Confianza", f"{confidence:.2f}%")
            st.write(f"**Salida del modelo:** {prediction_shape}")

    except Exception as e:
        st.error(f"Error procesando la imagen: {e}")

st.markdown("---")
st.caption("Aplicación desplegada con Streamlit.")