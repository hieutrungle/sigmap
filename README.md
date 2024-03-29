# SigMap

## Usage

Given a 3D mitsuba file format, this code computes the coverage maps of the structure

Moreover, it computes the optimal angle and position of the RIS elements to:

-   maximize the received power at a receiver
-   optimize the data rate

## Installation

Clone this repository and navigate to it in your terminal. Activate your environment. Then run:

```bash
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Install local sigmap package:

```bash
pip install -e .
```

## Prepare Blender with Mitsuba on Server without GUI

### Blender

-   Install Blender v3.3 on a local computer
-   Install [mitsuba-blender](https://github.com/mitsuba-renderer/mitsuba-blender) add-on
-   Install Blender v3.3 on a server without GUI
-   Copy local $(HOME)/.config/blender/3.3/config/userpref.blend to server $(HOME)/.config/blender/3.3/config/userpref.blend
-   Copy local $(HOME)/.config/blender/3.3/scripts/addons/mitsuba to server $(HOME)/.config/blender/3.3/scripts/addons/mitsuba

## Prepare data

### Folder structure

```bash
sigmap
├── sigmap
│   ├── data
├── assets
│   ├── blender
│   │   ├── scene_celing_color.blend
│   │   ├── scene_color.blend
│   ├── images
│   │   ├── scene_1
│   │   │   ├── image_conf_1.png
│   │   │   ├── image_conf_2.png
│   │   │   ├── image_conf_3.png
│   │   ├── scene_2
│   │   │   ├── image_conf_1.png
│   │   │   ├── image_conf_2.png
│   ├── videos
│   │   ├── scene_1.mp4
│   │   ├── scene_2.mp4

```

### Folder description

In `assets/blender` directory, you have to provide the following:

-   `scene_celing_color.blend`: Blender scene with the ceiling of the structure on. This file is used to compute coverage maps even though they are not visible using visualization tools.
-   `scene_color.blend`: Blender scene with the ceiling of the structure off so that coverage maps can be seen (visualization purposes)

The other two directories `assets/images` and `assets/videos` are used to store the images and videos of the structure. They will be automatically created when running the code.

## Run

```bash
bash ./run_main.sh
```

This script does the following:

-   Install all the required addons and python packages for Blender
-   Modify the base configuration file with the position of the receiver
-   Export blender file to mitsuba file format using a python script to configure RIS elements for beamforming
-   Compute coverage maps using the exported mitsuba file and the modified configuration file

<!-- 1. Export blender file to mitsuba file format

    ```bash
    BLENDER_DIR=${HOME}/blender
    SIGMAP_DIR=${HOME}/research/sigmap
    ASSETS_DIR=${SIGMAP_DIR}/assets

    echo Exporting coverage map...
    echo Blender directory: $BLENDER_DIR
    echo Coverage map directory: $SIGMAP_DIR
    echo -e Assets directory: $ASSETS_DIR '\n'

    ${BLENDER_DIR}/blender-3.3.12-linux-x64/blender \
        -b ${BLENDER_DIR}/models/simple_hallway_color.blend \
        --python ${SIGMAP_DIR}/sigmap/blender_script/hallway.py \
            -- -cfg ${SIGMAP_DIR}/config/simple_hallway.yaml -o ${ASSETS_DIR}/blender
    echo Done
    ```

2. Compute coverage maps

    ```bash
    python ./sigmap/main.py -cfg ./config/simple_hallway.yaml --verbose --video_enabled
    ``` -->

### Compute coverage maps

#### Use Automation Bash Script

```bash
bash ./run_main.sh
```

#### Run Manually

```bash
python sigmap/main.py --options
```

### Video Generation

```bash
ffmpeg -framerate 5 -i {PATH_TO_IMAGES}_%05d.png -r 30 -pix_fmt yuv420p {OUTPUT_VIDEO_PATH}.mp4
```

Example:

```bash
ffmpeg -framerate 5 -i ./images/tmp_beamfocusing_simple_hallway/hallway_%05d.png -r 30 -pix_fmt yuv420p ./videos/beamfocusing_fr_5.mp4
```

### Arguments

`main.py` has the following arguments:

-   `--config_file`, `-cfg`: Path to the configuration file
-   `--verbose`, `-v`: Print debug information
-   `--video_enabled`: Generate video
-   `--index`: temporary index to retrieve the correct blender scenes

Informaation for CLI arguments of blender scripts are located in [blender_script/README.md](./sigmap/blender_script/README.md)

## Features

-   [x] Coverage Map - Sionna
-   [x] Coverage Map Visualization - Sionna
-   [x] Coverage Map Animation - ffmpeg

## In Progress

-   [ ] Optimal Angle of RIS Elements to Maximize Received Power at a Receiver - Blender script
-   [ ] Optimal Position of RIS Elements to Adjust Delay in order to Have Constructive Interference at a Receiver - Blender script
-   [ ] Link Level Simulation - Sionna
-   [ ] Data Rate Optimization - Sionna
