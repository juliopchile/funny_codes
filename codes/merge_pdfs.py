import os
from typing import Iterable, Sequence, Tuple

from PyPDF2 import PdfMerger
from PyPDF2.pagerange import PageRange


def combine_pdfs_in_order(directory: str = "PDFS", output_filename: str = "PDF_combinado.pdf") -> str:
    """Combina todos los PDFs del directorio en orden alfabético y devuelve la ruta resultante."""
    pdf_files = sorted(
        (f for f in os.listdir(directory) if f.lower().endswith(".pdf")),
        key=str.lower,
    )
    if not pdf_files:
        raise ValueError(f"No se encontraron PDFs en {directory}")

    output_path = os.path.join(directory, output_filename)
    combine_pdf_slices(((os.path.join(directory, f), None, None) for f in pdf_files), output_path)
    return output_path


SlicePlan = Iterable[Tuple[str, int | None, int | None]]


def combine_pdf_slices(slice_plan: SlicePlan, output_path: str) -> str:
    """Combina segmentos de distintos PDFs definidos por rangos (start inclusive, end exclusive)."""
    merger = PdfMerger()
    try:
        for pdf_path, start, end in slice_plan:
            pages = _build_page_range(start, end)
            merger.append(pdf_path, pages=pages)
            print(f"Añadido segmento: {os.path.basename(pdf_path)} [{start}:{end}]")
        merger.write(output_path)
    finally:
        merger.close()
    return output_path


def _build_page_range(start: int | None, end: int | None) -> PageRange | None:
    """Convierte los límites en un PageRange entendido por PyPDF2."""
    if start is None and end is None:
        return None
    start_str = "" if start is None else str(start)
    end_str = "" if end is None else str(end)
    return PageRange(f"{start_str}:{end_str}")


if __name__ == "__main__":
    base_dir = "inputs"
    pdf1 = os.path.join(base_dir, "main.pdf")
    pdf2 = os.path.join(base_dir, "ConstanciaBiblioteca_firmado.pdf")

    demo_slices: Sequence[Tuple[str, int | None, int | None]] = (
        (pdf1, 0, 1),  # Primera página de PDF1
        (pdf2, 0, 1),  # Primera página de PDF2
        (pdf1, 1, None),  # Resto de las páginas de PDF1
    )

    salida_demo = os.path.join(base_dir, "PDF_demo.pdf")
    combine_pdf_slices(demo_slices, salida_demo)
    print(f"PDFs combinados en: {salida_demo}")
