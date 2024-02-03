#!/usr/bin/env bash

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

SIGMAP_DIR=$(dirname $SCRIPT_DIR) # Get the root directory of the project
ASSETS_DIR=${SIGMAP_DIR}/assets
# * Change this to your blender directory
BLENDER_DIR=${HOME}/blender 

echo Blender directory: $BLENDER_DIR
echo Coverage map directory: $SIGMAP_DIR
echo -e Assets directory: $ASSETS_DIR '\n'

BASE_CONFIG_FILE=${SIGMAP_DIR}/config/base_simple_hallway.yaml
WORK_CONFIG_FILE=${SIGMAP_DIR}/config/simple_hallway_tmp.yaml


xs=($(seq -15.0 1.0 -14.1))
ys=($(seq -2.5 -0.5 -3.0))
idx=0

# for x in ${xs[@]}; do
#     ys=( $(printf '%s\n' "${ys[@]}" | tac) )
#     for y in ${ys[@]}; do
#         echo "x: $x, y: $y"
        
#         # Modify configuration file
#         echo -e 'Modifying configuration file...'
#         python ${SIGMAP_DIR}/scripts/modify_config.py -cfg ${BASE_CONFIG_FILE} \
#             --output ${WORK_CONFIG_FILE} --rx_position " ${x}" " ${y}" 1.5
#         echo -e 'Finished modifying configuration file\n'

#         # Perform the export of the mitsuba scene using the blender script
#         echo -e 'Exporting mitsuba scene...'
#         ${BLENDER_DIR}/blender-3.3.12-linux-x64/blender \
#             -b ${BLENDER_DIR}/models/simple_hallway_color.blend \
#             --python ${SIGMAP_DIR}/sigmap/blender_script/hallway.py \
#                 -- -cfg ${WORK_CONFIG_FILE} -o ${ASSETS_DIR}/blender \
#                 --index ${idx}
#         echo -e 'Finished exporting mitsuba scene\n'

#         # Compute the coverage map
#         echo Compute coverage map...
#         python ${SIGMAP_DIR}/sigmap/main.py -cfg ${WORK_CONFIG_FILE}  \
#             --index ${idx} \
#             --verbose \
#             # --video_enabled
#         echo -e 'Finished computing coverage map\n'

#         idx=$((idx+1))
#     done
# done