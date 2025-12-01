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
from PIL import Image, ImageChops # Se importan librerías necesarias para el código

lista_archivos = os.listdir("/content/drive/MyDrive/Progra/test")
total_predicciones = 0
aciertos = 0
media_probabilidad_aciertos = 0.0
# predicciones incorrectas contendrá una lista con los nombres de las imágenes mal clasificadas
predicciones_incorrectas = []

def predecir_imagen(ruta_imagen):
    #Esto es el proceso que se utilizó en la fase 2, pero esta vez definido dentro de la función predecir_imagen
    imagen = Image.open(ruta_imagen).convert("RGB")

    #Esto es el proceso que se utilizó en la fase 3, pero esta vez definido dentro de la función predecir_imagen
    size = (224, 224)
    image = ImageOps.fit(imagen, size, Image.Resampling.LANCZOS)
    imagen_array = np.asarray(image)
    normalizada_imagen_array = (imagen_array.astype(np.float32) / 127.5) - 1

    #Esto es el proceso que se utilizó en la fase 4, pero esta vez definido dentro de la función predecir_imagen
    lote_imagenes = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    lote_imagenes[0] = normalizada_imagen_array
    resultados = mi_modelo.predict(lote_imagenes)

    #Esto es el proceso que se utilizó en la fase 5, pero esta vez definido dentro de la función predecir_imagen
    indice = np.argmax(resultados[0])
    etiqueta = nombre_clases[indice].strip() # Esto sirve para eliminar el carácter de nueva línea.
    probabilidad = resultados[0][indice]

    return etiqueta, probabilidad


for nombre_archivo in lista_archivos:
    if "cat" in nombre_archivo:
        etiqueta_esperada = "0 Cat"
    elif "dog" in nombre_archivo:
        etiqueta_esperada = "1 Dog"
    else:
        continue  # Saltar archivos que no sean de gatos o perros

    total_predicciones += 1

    ruta_imagen = os.path.join("/content/drive/MyDrive/Progra/test", nombre_archivo)

    # Aquí obtendriamos la predicción del modelo llamando a una función que
    # implemente las fases de inferencia
    etiqueta_predicha, probabilidad = predecir_imagen(ruta_imagen)

    if etiqueta_predicha == etiqueta_esperada:
        aciertos += 1
        media_probabilidad_aciertos += probabilidad
    else:
        info_error= {"archivo": nombre_archivo, "prediccion": etiqueta_predicha,
             "probabilidad": probabilidad}
        predicciones_incorrectas.append(info_error)



    # Cargamos el modelo
    modelo = keras.models.load_model("st-app/keras_model.h5", compile=False)
    # Carga las etiquetas de las clases
    clases = open("st-app/labels.txt", "r").readlines()
    return modelo, clases


# 1.CARGAMOS EL MODELO Y ETIQUETAS
try:
    mi_modelo, nombre_clases = carga_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()
# 2. CAPTURAMOS LA IMAGEN HACIENDO USO DE LA CÁMARA
imagen_camara = st.camera_input("Haz una foto")

# 3. PREDICCIÓN
if imagen_camara is not None:
    imagen = Image.open(imagen_camara).convert("RGB")
    imagen = ImageOps.fit(imagen, (224, 224), Image.Resampling.LANCZOS)
    imagen_array = np.asarray(imagen)
    normalizada_imagen_array = (imagen_array.astype(np.float32) / 127.5) - 1
# Crear un array para un lote de 1 imagen. ndarray = N-Dimensional Array
    lote_imagenes = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    lote_imagenes[0] = normalizada_imagen_array

    # Predicción
    resultados= mi_modelo.predict(lote_imagenes)
    indice = np.argmax(resultados[0])
    etiqueta = nombre_clases[indice]
    probabilidad = resultados[0][indice]

    st.divider() # Línea separadora visual

if "Perro" in etiqueta:
        st.success(f"¡Es un **PERRO**! 🐶")
        st.balloons() # Efecto visual
else:
        st.success(f"¡Es un **GATO**! 🐱")
        st.snow() # Efecto visual

st.write(f"Estoy un {probabilidad:.2%} seguro.")
