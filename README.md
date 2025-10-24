# funny_codes

`funny_codes` is a personal toolbox of one-off Python and FFmpeg helpers for media processing and file wrangling. Each script lives on its own and can be run directly once its dependencies are installed. This document gives you a high-level tour so you can quickly spot the utility you need and know how to kick it off.

## Getting Started

- Use Python 3.10+ if available.
- Create a virtual environment and activate it before installing packages.
- Install dependencies only for the scripts you plan to run. The table below lists the extra packages or tools each script expects.

### Dependency Cheatsheet

| Script | Purpose | Key Dependencies |
| --- | --- | --- |
| `codes/ansi2html.py` | Turn ANSI-colored logs into LaTeX with matching color definitions. | `ansi2html` |
| `codes/combine_images_to_pdf.py` | Stack multiple images into a single PDF. | `Pillow` |
| `codes/crop_image.py` | Interactive perspective crop of a document image. | `opencv-python`, `numpy` |
| `codes/crop_video.py` | Trim a video clip without re-encoding. | System `ffmpeg` |
| `codes/download_youtube_video.py` | Smart YouTube downloader that picks Resolve-friendly formats. | `yt-dlp`, `pandas`, `numpy`, system `ffmpeg` |
| `codes/extract_images_from_pdf.py` | Pull every embedded image out of a PDF. | `PyMuPDF` |
| `codes/ffmpeg_compatibility_test.py` | Benchmark FFmpeg encoder/muxer combinations and capture results. | `pandas`, `numpy`, system `ffmpeg` |
| `codes/improve_document_visibility.py` | Tune thresholds live and export a cleaned-up document PDF. | `opencv-python`, `numpy`, `Pillow` |
| `codes/merge_pdfs.py` | Merge all PDFs in a folder into one file. | `PyPDF2` |
| `codes/text_to_image.py` | Send an image edit request to OpenAI’s Images API. | `openai` plus API key |
| `codes/video_to_gif.py` | Render a short GIF from a video using a two-pass palette. | System `ffmpeg` |

> Tip: keep `ffmpeg` on your `PATH` so the video helpers can find it.

## Utility Notes

- `codes/ansi2html.py` reads an ANSI-formatted log (default `iperf3.txt`), converts it to LaTeX, and prints out `/definecolor` commands that match the ANSI palette.
- `codes/combine_images_to_pdf.py` expects you to edit the `imagenes` list and the output path before running `python codes/combine_images_to_pdf.py`.
- `codes/crop_image.py` opens an interactive window; click four corners, then press `c` to create `cropped_output.png` or `q` to quit.
- `codes/crop_video.py` wraps `ffmpeg -ss/-to/-t` to cut clips without transcoding. Supply start/end times or a duration.
- `codes/download_youtube_video.py` builds download plans from a playlist or single URL, skips already-downloaded items, and remuxes into containers verified as compatible with DaVinci Resolve (using data from `ffmpeg_test/ffmpeg_compatibility.json`).
- `codes/extract_images_from_pdf.py` can extract every image (`extract_all_images`) or just a single page (`extract_images_from_page`). Outputs go under `out/extracted_images/` by default.
- `codes/ffmpeg_compatibility_test.py` iterates through codec/muxer combos defined in `SCRIPT_CONFIG`, captures logs in `ffmpeg_test/bin/logs`, and updates lookup tables (`ffmpeg_compatibility.npy` + `.json`) for other tools.
- `codes/improve_document_visibility.py` denoises a document, lets you experiment with threshold sliders, and saves the result as `document_enhanced.pdf` when you press `p`.
- `codes/merge_pdfs.py` concatenates every `.pdf` in `PDFS/` (alphabetically) into `PDF_combinado.pdf`.
- `codes/text_to_image.py` calls OpenAI’s `images.edit` endpoint. Set your API key inside `codes/super_secrets.py` (`OPENAI_API_KEY = "your-key"`) and point `imagen` to the file you want to edit.
- `codes/video_to_gif.py` runs the common two-pass palette flow (`palettegen` + `paletteuse`) so GIF colors stay crisp.

## Project Layout

```
codes/        # Standalone Python utilities
ffmpeg_test/  # Artifacts and inputs for FFmpeg compatibility experiments
```
