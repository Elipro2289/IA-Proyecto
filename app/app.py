import os
# Esto obliga a TensorFlow a usar el modo compatibilidad con versiones antiguas
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
from tensorflow import keras 
from PIL import Image, ImageOps 
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Reconocimiento Perros vs Gatos", page_icon="🐾")

st.title("🐶 Detector de Mascotas 🐱")
st.write("Usa la cámara para saber si es un perro o un gato.")

# DEFINIMOS UNA FUNCIÓN PARA CARGAR EL MODELO Y GUARDARLO EN CACHE
# Usamos cache para que no se cargue cada vez que detecta un movimiento
@st.cache_resource
def carga_modelo():
    # Cargamos el modelo
    modelo = keras.models.load_model("app/keras_model.h5", compile=False)
    # Carga las etiquetas de las clases
    clases = open("app/labels.txt", "r").readlines()
    return modelo, clases


# 1.CARGAMOS EL MODELO Y ETIQUETAS
try:
    mi_modelo, nombre_clases = carga_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# 2. CAPTURAMOS LA IMAGEN HACIENDO USO DE LA CÁMARA
imagen_camara = st.camera_input("Haz una foto")

# 3. PREDICCIÓN Y RESULTADOS
if imagen_camara is not None:
    # 3.1 Procesamiento de la imagen
    imagen = Image.open(imagen_camara).convert("RGB")
    imagen = ImageOps.fit(imagen, (224, 224), Image.Resampling.LANCZOS)
    imagen_array = np.asarray(imagen)
    normalizada_imagen_array = (imagen_array.astype(np.float32) / 127.5) - 1
    
    # Crear un array para un lote de 1 imagen. ndarray = N-Dimensional Array
    lote_imagenes = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    lote_imagenes[0] = normalizada_imagen_array

    # 3.2 Predicción
    resultados = mi_modelo.predict(lote_imagenes)
    indice = np.argmax(resultados[0])
    etiqueta = nombre_clases[indice]
    probabilidad = resultados[0][indice]

    # 3.3 Mostrar resultados
    st.divider() # Línea separadora visual

    if "Dog" in etiqueta:
        st.success(f"¡Es un **PERRO**! 🐶")
        st.balloons() # Efecto visual
    else:
        st.success(f"¡Es un **GATO**! 🐱")
        st.snow() # Efecto visual
        
    st.write(f"Estoy un **{probabilidad:.2%}** seguro.") 
