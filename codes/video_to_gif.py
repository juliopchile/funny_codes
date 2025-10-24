# requirements ffmpeg installed
import subprocess
import os

def video_to_gif(input_path, output_path="output.gif", duration=8, width=480, fps=10):
    # Temporary palette image
    palette_path = "palette.png"

    # Step 1: Generate palette
    subprocess.run([
        "ffmpeg",
        "-y",
        "-t", str(duration),
        "-i", input_path,
        "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
        palette_path
    ], check=True)

    # Step 2: Use palette to generate gif
    subprocess.run([
        "ffmpeg",
        "-t", str(duration),
        "-i", input_path,
        "-i", palette_path,
        "-filter_complex", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
        output_path
    ], check=True)

    # Clean up palette
    os.remove(palette_path)

    print(f"GIF saved to {output_path}")

# Example usage
video_to_gif("Nogger Ice Cream Toffi.webm")
