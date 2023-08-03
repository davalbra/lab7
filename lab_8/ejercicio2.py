import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_shrink(image, Shrink_MIN, Shrink_MAX):
    image = image.astype(float)
    I_MIN = np.min(image.ravel())
    I_MAX = np.max(image.ravel())
    shrunk = Shrink_MIN + ((Shrink_MAX - Shrink_MIN) / (I_MAX - I_MIN)) * (image - I_MIN)
    return shrunk.astype(np.uint8)

def histogram_stretch(image, MIN=0, MAX=255):
    image = image.astype(float)
    I_MIN = np.min(image.ravel())
    I_MAX = np.max(image.ravel())
    image = (image - I_MIN) / (I_MAX - I_MIN)
    stretched = image * (MAX - MIN) + MIN
    return stretched.astype(np.uint8)

img = cv2.imread('example3.jpg', cv2.IMREAD_GRAYSCALE)

fft_img = np.fft.fft2(img)
img_fftshift = np.fft.fftshift(fft_img)

def filtro_pb(shape,cut_freq):
    rows,cols = shape
    center_r,center_c = rows//2, cols//2
    filter = np.zeros((rows,cols))
    filter[center_r- cut_freq:center_r + cut_freq,center_c -cut_freq:center_c + cut_freq]=1
    return filter

fshift = filtro_pb(img_fftshift.shape,40) * img_fftshift
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back_frecuen = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
imgCompress = np.clip(img-img_back_frecuen,0,255)
stretched = histogram_stretch(imgCompress)
result = np.add(img,stretched)
# Muestra todas las imágenes en una sola figura.
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Imagen original
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Imagen Original')

# Imagen en el dominio de frecuencia después del filtrado
axs[0, 1].imshow(np.log1p(np.abs(fshift)), cmap='gray')
axs[0, 1].set_title('Dominio de Frecuencia Filtrado')

# Imagen después del filtrado inverso de Fourier
axs[0, 2].imshow(img_back_frecuen, cmap='gray')
axs[0, 2].set_title('Inversa de Fourier')

# Imagen después del estiramiento del histograma
axs[1, 0].imshow(stretched, cmap='gray')
axs[1, 0].set_title('Estiramiento del Histograma')

# Imagen final
axs[1, 1].imshow(result, cmap='gray')
axs[1, 1].set_title('Imagen Resultante')

# Histograma final
bins_shrunk = np.linspace(0, 255, 255)
axs[1, 2].hist(result.ravel(), bins=bins_shrunk, color='black')
axs[1, 2].set_title('Histograma Comprimido')
axs[1, 2].set_xlabel('Valor de pixel')
axs[1, 2].set_ylabel('Frecuencia')

# Elimina los ejes vacíos
plt.tight_layout()
plt.show()
