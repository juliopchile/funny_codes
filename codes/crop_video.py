# requirements ffmpeg installed 

import subprocess
from pathlib import Path

def crop_video(input_path, output_path, start_time=None, end_time=None, duration=None):
    """
    Crop a video using ffmpeg.
    
    Args:
        input_path (str or Path): Path to the input video.
        output_path (str or Path): Path to save the cropped video.
        start_time (str or float, optional): Start time in seconds or hh:mm:ss format.
        end_time (str or float, optional): End time in seconds or hh:mm:ss format.
        duration (str or float, optional): Duration in seconds or hh:mm:ss format.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    ffmpeg_cmd = ["ffmpeg", "-y"]  # overwrite output

    if start_time:
        ffmpeg_cmd += ["-ss", str(start_time)]

    ffmpeg_cmd += ["-i", str(input_path)]

    # Priority: (end_time - start_time) > duration
    if end_time and start_time:
        ffmpeg_cmd += ["-to", str(end_time)]
    elif duration:
        ffmpeg_cmd += ["-t", str(duration)]

    ffmpeg_cmd += ["-c", "copy", str(output_path)]  # copy streams, no re-encoding

    print(f"Running command: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, check=True)


# ------------------------------
# 🧪 Example usage:

if __name__ == "__main__":
    crop_video(
        input_path="Numa Numa.mp4",
        output_path="output_cropped.mp4",
        start_time="00:00:01",     # or 5
        end_time="00:00:20",       # optional, used if you want time A to B
        duration=None              # or "10", used if end_time is not given
    )
