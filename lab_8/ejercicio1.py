import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_stretch(image, MIN=0, MAX=255):
    # Alargamiento de histogramas (stretch)
    image = image.astype(float)
    I_MIN = np.min(image.ravel())
    I_MAX = np.max(image.ravel())
    print (I_MIN)
    print (I_MAX)
    image = (image - I_MIN) / (I_MAX - I_MIN) # normalización al rango [0, 1]
    stretched = image * (MAX - MIN) + MIN  # estiramiento al rango [MIN, MAX]
    return stretched.astype(np.uint8)

def histogram_shrink(image, Shrink_MIN, Shrink_MAX):
    # Compresión de histogramas (shrink)
    # Shrink_MIN y Shrink_MAX son los límites del intervalo de compresión (0 <= Shrink_MIN < Shrink_MAX <= 255)
    image = image.astype(float)
    I_MIN = np.min(image.ravel())
    I_MAX = np.max(image.ravel())
    shrunk = Shrink_MIN + ((Shrink_MAX - Shrink_MIN) / (I_MAX - I_MIN)) * (image - I_MIN)
    return shrunk.astype(np.uint8)


def histogram_slide(image, d):
    # Desplazamiento de histogramas (slide)
    # d es la cantidad de desplazamiento (puede ser positiva o negativa)
    slid = image + d
    return np.clip(slid, 0, 255).astype(np.uint8)

# Cargar una imagen
image = cv2.imread('example3.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar los algoritmos
stretched = histogram_stretch(histogram_shrink(image, 50, 200))
shrunk = histogram_shrink(image, 50, 200)
slid = histogram_slide(image, 50)

# Mostrar las imágenes y los histogramas
fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
axs1[0].imshow(image, cmap='gray')
axs1[0].set_title('Imagen original')
axs1[1].hist(image.ravel(), bins=255, color='black')  # Reducir el número de bins
axs1[1].set_title('Histograma original')
axs1[1].set_xlabel('Valor de pixel')
axs1[1].set_ylabel('Frecuencia')

fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
axs2[0].imshow(stretched, cmap='gray')
axs2[0].set_title('Imagen estirada')
# Generar los bins en función del rango real de los valores de píxeles
bins_stretched = np.linspace(0, 255, 255)
# Generar el histograma
axs2[1].hist(stretched.ravel(), bins=bins_stretched, color='black')
axs2[1].set_title('Histograma estirado')
axs2[1].set_xlabel('Valor de pixel')
axs2[1].set_ylabel('Frecuencia')

fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))
axs3[0].imshow(shrunk, cmap='gray')
axs3[0].set_title('Imagen comprimida')
bins_shrunk = np.linspace(0, 255, 255)
axs3[1].hist(shrunk.ravel(), bins=bins_shrunk, color='black') 
axs3[1].set_title('Histograma comprimido')
axs3[1].set_xlabel('Valor de pixel')
axs3[1].set_ylabel('Frecuencia')

fig4, axs4 = plt.subplots(1, 2, figsize=(10, 5))
axs4[0].imshow(slid, cmap='gray')
axs4[0].set_title('Imagen desplazada')
bins_slid = np.linspace(0, 255, 255) # Se asume un rango completo 0-255 después del deslizamiento
axs4[1].hist(slid.ravel(), bins=bins_slid, color='black')
axs4[1].set_title('Histograma desplazado')
axs4[1].set_xlabel('Valor de pixel')
axs4[1].set_ylabel('Frecuencia')
plt.show()
