from sigmap.utils import logger, utils
import subprocess
import os


def check_ffmpeg_installed():
    try:
        subprocess.check_output(["which", "ffmpeg"])
        return True
    except subprocess.CalledProcessError:
        logger.log("ffmpeg is not installed. Please install ffmpeg to create videos.")
        return False


def check_if_images_exist(img_dir, config):
    criteria = [
        not os.path.exists(img_dir),
        not os.path.exists(
            os.path.join(img_dir, f"{config.mitsuba_filename}" + "_00000.png")
        ),
        len(os.listdir(img_dir)) == 0,
    ]
    if any(criteria):
        logger.log("No images found. Please run the simulation first.")
        return False
    else:
        return True


def create_video(img_dir, video_dir, config):
    if check_ffmpeg_installed() and check_if_images_exist(img_dir, config):
        video_path = utils.create_filename(video_dir, f"{config.scene_name}.mp4")
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "1",
                "-i",
                os.path.join(img_dir, f"{config.mitsuba_filename}" + "_%05d.png"),
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
        )
        logger.log(f"Video saved to {video_path}")
