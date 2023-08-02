import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_stretch(image, MIN=0, MAX=255):
    # Alargamiento de histogramas (stretch)
    I_MIN = np.min(image.ravel())
    I_MAX = np.max(image.ravel())
    stretched = ((image - I_MIN) * (MAX - MIN) / (I_MAX - I_MIN)) + MIN
    return stretched.astype(np.uint8)



def histogram_shrink(image, a, b):
    # Compresión de histogramas (shrink)
    # a y b son los límites del intervalo de compresión (0 <= a < b <= 255)
    shrunk = a + (b - a) * (image - np.min(image)) / (np.max(image) - np.min(image))
    return shrunk.astype(np.uint8)


def histogram_slide(image, d):
    # Desplazamiento de histogramas (slide)
    # d es la cantidad de desplazamiento (puede ser positiva o negativa)
    slid = image + d
    return np.clip(slid, 0, 255).astype(np.uint8)


# Cargar una imagen
image = cv2.imread('example3.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar los algoritmos
stretched = histogram_stretch(image)
shrunk = histogram_shrink(image, 50, 200)
slid = histogram_slide(image, -50)
# Mostrar las imágenes y los histogramas
fig, axs = plt.subplots(4, 2, figsize=(10, 20))

axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Imagen original')
axs[0, 1].hist(image.ravel(), bins=50, color='black')  # Reducir el número de bins
axs[0, 1].set_title('Histograma original')
axs[1, 0].imshow(stretched, cmap='gray')
axs[1, 0].set_title('Imagen estirada')
axs[1, 1].hist(stretched.ravel(), bins=50, color='black')  # Reducir el número de bins
axs[1, 1].set_title('Histograma estirado')

axs[2, 0].imshow(shrunk, cmap='gray')
axs[2, 0].set_title('Imagen comprimida')
axs[2, 1].hist(shrunk.ravel(), bins=50, color='black')  # Reducir el número de bins
axs[2, 1].set_title('Histograma comprimido')

axs[3, 0].imshow(slid, cmap='gray')
axs[3, 0].set_title('Imagen desplazada')
axs[3, 1].hist(slid.ravel(), bins=50, color='black')  # Reducir el número de bins
axs[3, 1].set_title('Histograma desplazado')

for ax in axs.flat:
    ax.label_outer()

plt.show()