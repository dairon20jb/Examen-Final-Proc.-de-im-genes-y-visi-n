from  Bandera import * # Importando
import os
import cv2
if __name__ == '__main__':
    path = 'C:\PRUEBA'
    image_name = 'flag2.png'
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    direccion = input('Digite la ruta de la imagen: ')  # Dirección de la carpeta en la que se encuentra la imagen
    nombre_imagen = input('Digite el nombre de la imagen: ')  # Nombre de la imagen a cargar, se debe incluir el formato
    Info_Bandera = Bandera(path, image_name)  # LLamado clase
    Info_Bandera.Colores()
    Info_Bandera.Porcentaje()
    Orientacion=Info_Bandera.Orientacion()
    print(Orientacion)



