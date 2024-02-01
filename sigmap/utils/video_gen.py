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


def check_if_images_exist(img_tmp_folder, config):
    criteria = [
        not os.path.exists(img_tmp_folder),
        not os.path.exists(
            os.path.join(img_tmp_folder, f"{config.mitsuba_filename}" + "_00.png")
        ),
        len(os.listdir(img_tmp_folder)) == 0,
    ]
    if any(criteria):
        logger.log("No images found. Please run the simulation first.")
        return False
    else:
        return True


def create_video(img_tmp_folder, video_folder, config):
    if check_ffmpeg_installed() and check_if_images_exist(img_tmp_folder, config):
        video_path = utils.create_filename(video_folder, f"{config.scene_name}.mp4")
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "1",
                "-i",
                os.path.join(
                    img_tmp_folder, f"{config.mitsuba_filename}" + "_%02d.png"
                ),
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ]
        )
        logger.log(f"Video saved to {video_path}")
