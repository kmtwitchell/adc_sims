import numpy as np 
import matplotlib.pyplot as plt
from hcipy import *
import os
from adcFitter_onsky import AdcFitter

folder_path = '/Users/Katie/Documents/MagAO-X/Data/ADC/2025B/onsky_obs/2025-11-30_042051_tau_ceti_adc_tests_offset/camsci1'

# Get all image file paths in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("camsci1") and file.endswith('.fits')]
image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))


# Load all images into a list
images = [read_fits(img_path) for img_path in image_files]

print(f"Loaded {len(images)} images.")

# #system params
# wavelength = 764E-9
# bw = 120E-9
# grating_angle = 28
# grating_freq = 47

# ADC = AdcFitter(wavelength=wavelength,grating_freq=grating_freq)

#reference cube for the extent. make sure this is in the same wavelength as your data
# difference_cube = read_field('/Users/Katie/Documents/Fall 2023/MagAO-X/ADC Sim code/updated onsky/all_lab_fields_i.fits')
# igrid = difference_cube.grid
# print(igrid.dims)
# print((images[0].size))

img = images[0]
dim = np.sqrt(img.size)
extent = dim * 6/21
pgrid = make_pupil_grid(dim,extent)
img = Field(img.ravel(),pgrid)

# plt.figure()
# imshow_field(img)
# plt.colorbar()
# plt.show()

fields = []

for img in images:
    im = img
    im = im.ravel()
    im = Field(im,pgrid)
    fields.append(im)

fieldcube = Field(fields,fields[-1].grid)
write_field(fieldcube,'/Users/Katie/Documents/MagAO-X/simulations/adc_sims/onsky_25b/fields/tau_ceti_adc_tests_offset_i.fits')