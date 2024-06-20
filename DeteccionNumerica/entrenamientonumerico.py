import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar los datos de MNIST
(X_entrenamiento, Y_entrenamiento), (X_pruebas, Y_pruebas) = mnist.load_data()

# Colocar los datos en la forma correcta (1, 28, 28, 1)
X_entrenamiento = X_entrenamiento.reshape(X_entrenamiento.shape[0], 28, 28, 1)
X_pruebas = X_pruebas.reshape(X_pruebas.shape[0], 28, 28, 1)

# Hacer 'one-hot encoding' de los resultados
Y_entrenamiento = to_categorical(Y_entrenamiento)
Y_pruebas = to_categorical(Y_pruebas)

# Convertir a flotante y normalizar
X_entrenamiento = X_entrenamiento.astype('float32') / 255
X_pruebas = X_pruebas.astype('float32') / 255

#Codigo para mostrar imagenes del set, no es necesario ejecutarlo, solo imprime unos numeros :)
filas = 2
columnas = 8
num = filas*columnas
imagenes = X_entrenamiento[0:num]
etiquetas = Y_entrenamiento[0:num]
fig, axes = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for i in range(num):
     ax = axes[i//columnas, i%columnas]
     ax.imshow(imagenes[i].reshape(28,28), cmap='gray_r')
     ax.set_title('Label: {}'.format(np.argmax(etiquetas[i])))
plt.tight_layout()
plt.show()

#Aumento de datos
#Variables para controlar las transformaciones que se haran en el aumento de datos
#utilizando ImageDataGenerator de keras
rango_rotacion = 30
mov_ancho = 0.25
mov_alto = 0.25
rango_acercamiento = [0.5, 1.5]

datagen = ImageDataGenerator(
    rotation_range=rango_rotacion,
    width_shift_range=mov_ancho,
    height_shift_range=mov_alto,
    zoom_range=rango_acercamiento,
)

datagen.fit(X_entrenamiento)

#Codigo para mostrar imagenes del set, no es necesario ejecutarlo, solo imprime como se ven antes y despues de las transformaciones
filas = 4
columnas = 8
num = filas*columnas
print('ANTES:\n')
fig1, axes1 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for i in range(num):
     ax = axes1[i//columnas, i%columnas]
     ax.imshow(X_entrenamiento[i].reshape(28,28), cmap='gray_r')
     ax.set_title('Label: {}'.format(np.argmax(Y_entrenamiento[i])))
plt.tight_layout()
plt.show()
print('DESPUES:\n')
fig2, axes2 = plt.subplots(filas, columnas, figsize=(1.5*columnas,2*filas))
for X, Y in datagen.flow(X_entrenamiento,Y_entrenamiento.reshape(Y_entrenamiento.shape[0], 10),batch_size=num,shuffle=False):
     for i in range(0, num):
          ax = axes2[i//columnas, i%columnas]
          ax.imshow(X[i].reshape(28,28), cmap='gray_r')
          ax.set_title('Label: {}'.format(int(np.argmax(Y[i]))))
     break
plt.tight_layout()
plt.show()

# Modelo
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compilación
modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Imprimir el resumen del modelo para verificar la correcta inicialización
modelo.summary()

# Los datos para entrenar saldran del datagen, de manera que sean generados con las transformaciones que indicamos
data_gen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=32)

TAMANO_LOTE = 32
# Calcular steps_per_epoch y validation_steps
steps_per_epoch = len(X_entrenamiento) // TAMANO_LOTE
validation_steps = len(X_pruebas) // TAMANO_LOTE



# Crear un dataset repetitivo para asegurar suficientes datos
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_gen_entrenamiento,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, 28, 28, 1], [None, 10])
).repeat()

validation_dataset = tf.data.Dataset.from_tensor_slices((X_pruebas, Y_pruebas)).batch(TAMANO_LOTE).repeat()

# Entrenar la red
print("Entrenando modelo...")
epocas = 5
try:
    history = modelo.fit(
        train_dataset,
        epochs=epocas,
        validation_data=validation_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    print("Modelo entrenado!")
except Exception as e:
    print(f"Ocurrió un error durante el entrenamiento: {e}")

# Exportar el modelo en el nuevo formato recomendado
modelo.save('numeros_conv_ad_do.h5')
