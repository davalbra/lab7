import cv2
import numpy as np

# Leer la imagen a color
img = cv2.imread('images_deber_lab7/deber#1_ruido_periodico.jpg', cv2.IMREAD_COLOR)

# Verifica que la imagen se haya cargado correctamente
if img is None:
    print("Error: la imagen no se pudo cargar. Asegúrate de que la ruta a la imagen es correcta.")
    exit()

# Definir el factor de escala y el tamaño de la imagen escalada
scale_factor = 2.0
new_size = (int(img.shape[1]*scale_factor), int(img.shape[0]*scale_factor))

# Crear una imagen vacía para almacenar el resultado
img_back = np.zeros(img.shape, np.float32)

# Aplicar la transformada de Fourier, la máscara y la transformada inversa a cada canal de color
for i in range(3):
    # Convertir la imagen a float32
    img_float32 = np.float32(img[:,:,i])

    # Aplicar la Transformada de Fourier usando la función dft()
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)

    # Mover el cero de las frecuencias al centro usando la función shift
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape[:2]
    crow, ccol = int(rows/2), int(cols/2) # centro de la imagen

    # Supongamos que el ruido se sitúa en dos líneas horizontales a y b de la imagen
    mask = np.ones((rows, cols, 2), np.float32) # inicializar la máscara como una matriz de unos

    # bloquear las líneas de ruido
    width = 10 # ancho de la línea de ruido

    # asumiendo que 'a' y 'b' son las posiciones de las líneas de ruido
    mask[:,80:90] = 0
    mask[:,170:180] = 0

    # Aplicar la máscara al dominio de la frecuencia
    fshift = dft_shift * mask

    # Obtener el espectro de magnitud y aplicar logaritmo para reducir la alta dinámica
    mask =cv2.magnitude(mask[:,:,0], mask[:,:,1])
    #espectro con mascara
    magnitude_spectrumWithMask = cv2.magnitude(fshift[:,:,0], fshift[:,:,1])
    magnitude_spectrumWithMask += 1 # Evitar el logaritmo de cero
    magnitude_spectrumWithMask = np.log(magnitude_spectrumWithMask)
    # Espectro de magnitud sin máscara
    magnitude_spectrumWithOut = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    magnitude_spectrumWithOut += 1 # Evitar el logaritmo de cero
    magnitude_spectrumWithOut = np.log(magnitude_spectrumWithOut)
    # Normalizar el espectro de magnitud para que esté entre 0 y 255 y convertirlo a uint8
    magnitude_spectrumWithOut = cv2.normalize(magnitude_spectrumWithOut, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Invertir los colores del espectro de magnitud
    magnitude_spectrumWithOut = 255 - magnitude_spectrumWithOut
    # Normalizar el espectro de magnitud para que esté entre 0 y 255 y convertirlo a uint8
    magnitude_spectrumWithMask = cv2.normalize(magnitude_spectrumWithMask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Invertir los colores del espectro de magnitud
    magnitude_spectrumWithMask = 255 - magnitude_spectrumWithMask

    # Volver al dominio del espacio utilizando la Transformada Inversa de Fourier
    f_ishift = np.fft.ifftshift(fshift)
    img_back_channel = cv2.idft(f_ishift)
    img_back_channel = cv2.magnitude(img_back_channel[:,:,0], img_back_channel[:,:,1])

    # Convertir la imagen de salida a un rango de 0 a 255
    img_back_channel = cv2.normalize(img_back_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Almacenar el canal procesado en la imagen de salida
    img_back[:,:,i] = img_back_channel

# Convertir la imagen de salida a uint8
img_back = cv2.convertScaleAbs(img_back)
# Escalar la imagen original y la imagen de salida
img = cv2.resize(img, new_size)
espectrum= cv2.resize(magnitude_spectrumWithMask, new_size)
frecuencia= cv2.resize(magnitude_spectrumWithOut, new_size)
mascara = cv2.resize(mask, new_size)
back = cv2.resize(img_back, new_size)
# Mostrar la imagen original y la imagen después de eliminar el ruido
cv2.imshow('Imagen Original', img)
cv2.imshow('Espectro de magnitud', espectrum)
cv2.imshow('Máscara', mask)
cv2.imshow('Frecuencia', frecuencia)
cv2.imshow('Máscara', mascara)
cv2.imshow('Imagen después de eliminar el ruido', back)

cv2.waitKey()
cv2.destroyAllWindows()
