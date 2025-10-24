import os
from PyPDF2 import PdfMerger

# Ruta del directorio donde están los PDFs
directorio = "PDFS"

# Nombre del PDF combinado
salida = "PDF_combinado.pdf"

# Crear el objeto que unirá los PDFs
merger = PdfMerger()

# Obtener lista de archivos PDF en el directorio
archivos_pdf = [f for f in os.listdir(directorio) if f.lower().endswith(".pdf")]
archivos_pdf.sort()  # Opcional: asegura un orden alfabético

# Agregar cada PDF al merger
for archivo in archivos_pdf:
    ruta_archivo = os.path.join(directorio, archivo)
    merger.append(ruta_archivo)
    print(f"Añadido: {archivo}")

# Guardar el archivo combinado
merger.write(os.path.join(directorio, salida))
merger.close()

print(f"PDFs combinados en: {salida}")
