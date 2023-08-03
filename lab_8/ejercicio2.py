import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_shrink(image, Shrink_MIN, Shrink_MAX):
    # Compresión de histogramas (shrink)
    # Shrink_MIN y Shrink_MAX son los límites del intervalo de compresión (0 <= Shrink_MIN < Shrink_MAX <= 255)
    image = image.astype(float)
    I_MIN = np.min(image.ravel())
    I_MAX = np.max(image.ravel())
    shrunk = Shrink_MIN + ((Shrink_MAX - Shrink_MIN) / (I_MAX - I_MIN)) * (image - I_MIN)
    return shrunk.astype(np.uint8)

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
# Cargar la imagen en escala de grises
img = cv2.imread('example3.jpg', cv2.IMREAD_GRAYSCALE)
scale_factor = 0.5
new_size = (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor))

# Obtener la Transformada de Fourier
fft_img = np.fft.fft2(img)
img_fftshift = np.fft.fftshift(fft_img)

# Crear un filtro paso bajo

def filtro_pb(img,cut_freq):
    arr=np.copy(img)
    rows,cols=arr.shape
    center_r,center_c=rows//2, cols//2
    filter=np.zeros((rows,cols), np.uint8)
    filter[center_r- cut_freq:center_r + cut_freq,center_c -cut_freq:center_c + cut_freq]=1
    return arr * filter
fshift= filtro_pb(img_fftshift,90)


# Obtener la imagen filtrada
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
img_back_frecuen = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
#shrunk = histogram_shrink(img_back_frecuen, 0, 170)
imgCompress = np.clip(img-img_back_frecuen,0,255) 
stretched = histogram_stretch(imgCompress)
result =np.add(img,stretched)


#aplicando shrunk

fig3, axs3 = plt.subplots(1, 2, figsize=(10, 5))
axs3[0].imshow(result, cmap='gray')
axs3[0].set_title('Imagen resultante')
bins_shrunk = np.linspace(0, 255, 255)
axs3[1].hist(result.ravel(), bins=bins_shrunk, color='black') 
axs3[1].set_title('Histograma comprimido')
axs3[1].set_xlabel('Valor de pixel')
axs3[1].set_ylabel('Frecuencia')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()