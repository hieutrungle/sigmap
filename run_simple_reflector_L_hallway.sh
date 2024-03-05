#!/usr/bin/env bash

set -o pipefail # Pipe fails when any command in the pipe fails
set -u  # Treat unset variables as an error

handle_error() {
    echo "An error occurred on line $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# # Source: https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
# # Get the directory of the script (does not solve symlink problem)
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# echo "Script directory: $SCRIPT_DIR"

# Get the source path of the script, even if it's called from a symlink
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
echo "Source directory: $SCRIPT_DIR"
SIGMAP_DIR=$SCRIPT_DIR
ASSETS_DIR=${SIGMAP_DIR}/assets

# * Change this to your blender directory
RESEARCH_DIR=$(dirname $SIGMAP_DIR)
HOME_DIR=$(dirname $RESEARCH_DIR)
BLENDER_DIR=${HOME_DIR}/blender 

echo Blender directory: $BLENDER_DIR
echo Coverage map directory: $SIGMAP_DIR
echo -e Assets directory: $ASSETS_DIR '\n'

BASE_CONFIG_FILE=${SIGMAP_DIR}/config/simple_reflector_L_hallway.yaml
filename=$(basename -- "$BASE_CONFIG_FILE")
WORK_CONFIG_FILE=${SIGMAP_DIR}/config/tmp_${filename}

# Find the blender executable
for file in ${BLENDER_DIR}/*
do
    if [[ "$file" == *"blender-3.3"* ]];then
        BLENDER_APP=$file/blender
    fi
done

# Open a random blender file to install and enable the mitsuba plugin
mkdir -p ${BLENDER_DIR}/addons
if [ ! -f ${BLENDER_DIR}/addons/mitsuba*.zip ]; then
    wget -P ${BLENDER_DIR}/addons https://github.com/mitsuba-renderer/mitsuba-blender/releases/download/v0.3.0/mitsuba-blender.zip 
    # unzip mitsuba-blender.zip -d ${BLENDER_DIR}/addons
fi
${BLENDER_APP} -b ${BLENDER_DIR}/models/simple_hallway_color.blend --python ${SIGMAP_DIR}/sigmap/blender_script/install_mitsuba_addon.py

# get scene_name from BASE_CONFIG_FILE
SCENE_NAME=$(python -c "import yaml; print(yaml.safe_load(open('${BASE_CONFIG_FILE}', 'r'))['scene_name'])")

# Setting up the environment
num_samples=10e6

# Main loop to compute the coverage map
xs=($(seq -3.0 0.25 -2.0))
ys=($(seq -2.5 -0.25 -4.5))
idx=0
for x in ${xs[@]}; do
    ys=( $(printf '%s\n' "${ys[@]}" | tac) )
    for y in ${ys[@]}; do
        idx_str=$(printf "%05d" ${idx})
        echo "x: $x, y: $y"
        
        # Modify configuration file
        echo -e 'Modifying configuration file...'
        python ${SIGMAP_DIR}/sigmap/blender_script/modify_config.py -cfg ${BASE_CONFIG_FILE} \
            --output ${WORK_CONFIG_FILE} --rx_position " ${x}" " ${y}" 1.5 --num_samples "${num_samples}"
        echo -e 'Finished modifying configuration file\n'

        # Perform the export of the mitsuba scene using the blender script
        echo -e 'Exporting mitsuba scene...'
        ${BLENDER_APP} \
            -b ${BLENDER_DIR}/models/simple_hallway_color.blend \
            --python ${SIGMAP_DIR}/sigmap/blender_script/hallway.py \
                -- -cfg ${WORK_CONFIG_FILE} -o ${ASSETS_DIR}/blender \
                --index ${idx_str}
        echo -e 'Finished exporting mitsuba scene\n'

        # Find the scene path with correct index
        COMPUTE_SCENE_PATH=$(find ${ASSETS_DIR}/blender/${SCENE_NAME} -type d -name "ceiling_idx_${idx_str}*")
        COMPUTE_SCENE_PATH=$(find "${COMPUTE_SCENE_PATH}" -type f -name "*.xml")
        VIZ_SCENE_PATH=$(find ${ASSETS_DIR}/blender/${SCENE_NAME} -type d -name "idx_${idx_str}*")
        VIZ_SCENE_PATH=$(find "${VIZ_SCENE_PATH}" -type f -name "*.xml")
        echo "Compute scene path: $COMPUTE_SCENE_PATH"
        echo "Viz scene path: $VIZ_SCENE_PATH"

        # Compute the coverage map
        echo Compute coverage map...
        python ${SIGMAP_DIR}/sigmap/main.py -cfg ${WORK_CONFIG_FILE}  \
            --compute_scene_path "${COMPUTE_SCENE_PATH}" \
            --viz_scene_path "${VIZ_SCENE_PATH}" \
            --cmap_enabled \
            --verbose
            # --video_enabled
        echo -e 'Finished computing coverage map\n'

        # # Clean up generated files
        # echo Cleaning up generated files...
        # rm -rf ${ASSETS_DIR}/blender
        # echo -e 'Finished cleaning up generated files\n'

        idx=$((idx+1))
    done
done