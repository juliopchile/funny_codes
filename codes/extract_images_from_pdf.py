# requeriments PyMuPDF
# pip install PyMuPDF

import fitz  # PyMuPDF
import os


def extract_all_images(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        
        # Crear subdirectorio por página
        page_dir = os.path.join(output_dir, f"page{page_index + 1}")
        os.makedirs(page_dir, exist_ok=True)
        
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            filename = f"{page_dir}/img{img_idx + 1}.{image_ext}"
            with open(filename, "wb") as f:
                f.write(image_bytes)

def extract_images_from_page(pdf_path, page_number, output_dir):
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    images = page.get_images(full=True)
    
    # Crear subdirectorio para la página específica
    page_dir = os.path.join(output_dir, f"page{page_number}")
    os.makedirs(page_dir, exist_ok=True)
    
    for img_idx, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        filename = f"{page_dir}/img{img_idx + 1}.{image_ext}"
        with open(filename, "wb") as f:
            f.write(image_bytes)


if __name__ == "__main__":
    pdf_path = "inputs/FormularioTrabajoTitulo_JulioLopez_firmado.pdf"
    output_dir = "out/extracted_images"
    #page_number = 14

    #extract_images_from_page(pdf_path, page_number, output_dir)
    extract_all_images(pdf_path, output_dir)
