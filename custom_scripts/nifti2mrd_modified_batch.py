import subprocess

files = []

output_file = "data/dcm2niix_converts/input.h5"
basepath= "data/dcm2niix_converts/"
for file in files:
    subprocess.run(["python3","custom_scripts/nifti2mrd_modified.py","-i",basepath+file,"-o",output_file])
print(f"✅ {len(files)} Converted!")
