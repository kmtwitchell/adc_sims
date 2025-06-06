import numpy as np 
import matplotlib.pyplot as plt
from hcipy import *
import os
from adcFitter_onsky import AdcFitter

folder_path = '/Users/Katie/Documents/MagAO-X/Data/ADC/lab_625/2025-06-02_235649_adc_6225_0offset/camsci1'

# Get all image file paths in the folder
image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.startswith("camsci1_") and file.endswith('.fits')]
image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))


# Load all images into a list
images = [read_fits(img_path) for img_path in image_files]

print(f"Loaded {len(images)} images.")

#system params
wavelength = 764E-9
bw = 120E-9
grating_angle = 28
grating_freq = 47

ADC = AdcFitter(wavelength=wavelength,grating_freq=grating_freq)

#reference cube for the extent. make sure this is in the same wavelength as your data
difference_cube = read_field('/Users/Katie/Documents/Fall 2023/MagAO-X/ADC Sim code/updated onsky/all_lab_fields_i.fits')
igrid = difference_cube.grid
print(igrid.dims)
print((images[0].size))

extent = np.array(igrid.separated_coords).max() * 2
grid = make_uniform_grid([512,512],extent)
fields = []

for img in images:
    im = img
    im = im.ravel()
    im = Field(im,grid)
    fields.append(im)

# plt.figure()
# imshow_field(fields[0])
# plt.show()

fieldcube = Field(fields,fields[-1].grid)
write_field(fieldcube,'/Users/Katie/Documents/MagAO-X/simulations/adc_sims/lab_6225_0offset.fits')