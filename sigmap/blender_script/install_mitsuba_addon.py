import addon_utils
import os
import bpy
import subprocess
import glob

current_dir = os.path.dirname(os.path.realpath(__file__))
sigmap_dir = os.path.dirname(os.path.dirname(current_dir))
research_dir = os.path.dirname(sigmap_dir)
scratch_dir = os.path.dirname(research_dir)
blender_dir = os.path.join(scratch_dir, "blender")
addon_dir = os.path.join(blender_dir, "addons")

# Define path to your downloaded script
path_to_script_dir = addon_dir

# Define a list of the files in this folder, i.e. directory. The method listdir() will return this list from our folder of downloaded scripts.
file_list = sorted(os.listdir(path_to_script_dir))

# Further specificy that of this list of files, you only want the ones with the .zip extension.
script_list = [item for item in file_list if item.endswith(".zip")]

# Specify the file path of the individual scripts (their names joined with the location of your downloaded scripts folder) then use wm.addon_install() to install them.
for file in file_list:
    path_to_file = os.path.join(path_to_script_dir, file)
    bpy.ops.preferences.addon_install(
        overwrite=True,
        target="DEFAULT",
        filepath=path_to_file,
        filter_folder=True,
        filter_python=False,
        filter_glob="*.py;*.zip",
    )

# install mitsuba using blender python using subprocess
blender_version = "3.3"
blender_folder = glob.glob(f"{blender_dir}/blender-{blender_version}*")[0]
python_exec = os.path.join(blender_folder, blender_version, "python", "bin")
python_exec = glob.glob(os.path.join(python_exec, "python3*"))[0]
# print(f"python_exec: {python_exec}")
subprocess.run([python_exec, "-m", "ensurepip"])
subprocess.run([python_exec, "-m", "pip", "install", "--upgrade", "pip"])
subprocess.run([python_exec, "-m", "pip", "install", "mitsuba"])
subprocess.run([python_exec, "-m", "pip", "install", "PyYAML"])

# Specify which add-ons you want enabled. For example, Crowd Render, Pie Menu Editor, etc. Use the script's python module.
enableTheseAddons = ["mitsuba-blender"]

# Use addon_enable() to enable them.
for string in enableTheseAddons:
    name = enableTheseAddons
    addon_utils.enable(string)
