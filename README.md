# SigMap

## Usage

Given a 3D mitsuba file format, this code computes the coverage maps of the structure

Moreover, it computes the optimal angle and position of the RIS elements to:

- maximize the received power at a receiver
- optimize the data rate

## Installation

Clone this repository and navigate to it in your terminal. Activate your environment. Then run:

```bash
pip install -r requirements.txt
```

Install local sigmap package:

```bash
pip install -e .
```

## Prepare data

### Folder structure

``` bash
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

- `scene_celing_color.blend`: Blender scene with the ceiling of the structure on. This file is used to compute coverage maps even though they are not visible using visualization tools.
- `scene_color.blend`: Blender scene with the ceiling of the structure off so that coverage maps can be seen (visualization purposes)

The other two directories `assets/images` and `assets/videos` are used to store the images and videos of the structure. They will be automatically created when running the code.

## Run

### Compute coverage maps

```bash
python sigmap/main.py 
```

### Arguments

[TODO] Add arguments

## Features

- [x] Coverage Map - Sionna
- [x] Coverage Map Visualization - Sionna
- [x] Coverage Map Animation - ffmpeg

## In Progress

- [ ] Optimal Angle of RIS Elements to Maximize Received Power at a Receiver - Blender script
- [ ] Optimal Position of RIS Elements to Adjust Delay in order to Have Constructive Interference at a Receiver - Blender script
- [ ] Link Level Simulation - Sionna
- [ ] Data Rate Optimization - Sionna
