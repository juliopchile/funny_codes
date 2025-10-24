import os
from PIL import Image

def imagenes_a_pdf(lista_imagenes, archivo_salida):
    """
    Convierte una lista de imágenes en un PDF.
    
    :param lista_imagenes: Lista con rutas a las imágenes.
    :param archivo_salida: Nombre del archivo PDF resultante.
    """
    # Abrir todas las imágenes y convertirlas a RGB
    imagenes = [Image.open(img).convert("RGB") for img in lista_imagenes]

    # Asegurar que el directorio de salida exista
    directorio_salida = os.path.dirname(archivo_salida)
    if directorio_salida:
        os.makedirs(directorio_salida, exist_ok=True)

    # Guardar como PDF (la primera imagen y el resto como páginas adicionales)
    if imagenes:
        imagenes[0].save(archivo_salida, save_all=True, append_images=imagenes[1:])
        print(f"PDF creado con éxito: {archivo_salida}")
    else:
        print("No se encontraron imágenes para convertir.")

# Ejemplo de uso
imagenes = ["inputs/1.jpg", "inputs/2.jpg", "inputs/3.jpg"]
imagenes_a_pdf(imagenes, "out/pdf/reembolso_médico.pdf")
