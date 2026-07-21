import numpy as np 
import matplotlib.pyplot as plt
from hcipy import *
import os

folder_path = '/Users/Katie/Documents/MagAO-X/Data/ADC/2026A/2026-03-31_072854_alfCir_adc_orientation_control/camsci1'

# Get all image file paths in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("camsci1") and file.endswith('.fits')]
image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))


# Load all images into a list
images = [read_fits(img_path) for img_path in image_files]

print(f"Loaded {len(images)} images.")

img = images[0]
dim = np.sqrt(img.size)
extent = dim * 6/21
pgrid = make_pupil_grid(dim,extent)
img = Field(img.ravel(),pgrid)

fields = []

for img in images[0:200]:
    im = img
    im = im.ravel()
    im = Field(im,pgrid)
    fields.append(im)

fieldcube = Field(fields,fields[-1].grid)
write_field(fieldcube,'/Users/Katie/Documents/MagAO-X/Data/ADC/2026A/fieldcube_orientation.fits')