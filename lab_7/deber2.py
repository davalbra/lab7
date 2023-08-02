import cv2
import numpy as np

# Leer la imagen a color
img = cv2.imread('images_deber_lab7/deber#2_img_trasladada.png', cv2.IMREAD_GRAYSCALE)

# Verifica que la imagen se haya cargado correctamente
if img is None:
    print("Error: la imagen no se pudo cargar. Asegúrate de que la ruta a la imagen es correcta.")
    exit()

# Convertir la imagen a float32
img_float32 = np.float32(img)

# Aplicar la Transformada de Fourier usando la función dft()
dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)

# Mover el cero de las frecuencias al centro usando la función shift
dft_shift = np.fft.fftshift(dft)

# Obtener el espectro de magnitud y aplicar logaritmo para reducir la alta dinámica
magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
magnitude_spectrum += 1 # Evitar el logaritmo de cero
magnitude_spectrum = np.log(magnitude_spectrum)

# Normalizar el espectro de magnitud para que esté entre 0 y 255 y convertirlo a uint8
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Invertir los colores del espectro de magnitud
magnitude_spectrum = 255 - magnitude_spectrum

# Definir el factor de escala y el tamaño de la imagen escalada
scale_factor = 2.0
new_size = (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor))

# Escalar la imagen original y el espectro de magnitud
img = cv2.resize(img, new_size)
magnitude_spectrum = cv2.resize(magnitude_spectrum, new_size)

# Mostrar la imagen original y el espectro de magnitud
cv2.imshow('Imagen Original', img)
cv2.imshow('Espectro de Magnitud', magnitude_spectrum)

cv2.waitKey()
cv2.destroyAllWindows()