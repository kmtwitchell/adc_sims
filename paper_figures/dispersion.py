import sys
import os

import numpy as np
from hcipy import *
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname('adc_sims'), 'simulations')))
from psfGenerator import PsfGenerator

wavelength = 764E-9
wavelength_normalized = wavelength / 656E-9
grating_angle = 28
grating_frequency = 30

mono_bw = 1E-9
bb_bw = 120E-9

#making the figure for the paper (redoing the one from the SPIE proceeding)
color = 'viridis'

PsfGen_mono = PsfGenerator(center_wavelength=wavelength_normalized,bandwidth=mono_bw/wavelength, focal_sampling=6,focal_extent = grating_frequency*1.25, grating_angle=grating_angle, fx=grating_frequency, fy = grating_frequency, mask=False)
PsfGen_broad = PsfGenerator(center_wavelength=wavelength_normalized,bandwidth=bb_bw/wavelength, focal_sampling=6,focal_extent = grating_frequency*1.25,grating_angle=grating_angle, fx=grating_frequency, fy = grating_frequency, mask=False)

psf_mono = PsfGen_mono.psf()
psf_bb = PsfGen_broad.psf()
psf_disp = PsfGen_broad.psf(magnitude_x = 0,magnitude_y = 0.1)

# center_of_intensity = np.array([sum(psf_bb*psf_bb.grid.x)/sum(psf_bb),sum(psf_bb*psf_bb.grid.y)/sum(psf_bb)])
# mask_ap = make_circular_aperture(30,center_of_intensity)
# mask = mask_ap(psf_bb.grid)
# mask = abs(mask - 1)
# psf_bb = mask * psf_bb
# psf_disp = mask * psf_disp

psf_mono = np.log10(PsfGen_mono.psf()) 
psf_bb = np.log10(PsfGen_broad.psf())
psf_disp = np.log10(PsfGen_broad.psf(magnitude_x = 0,magnitude_y = 0.1))

# psf_mono /= max(psf_mono)
# psf_bb /= max(psf_bb)
# psf_disp /= max(psf_disp)

#fig, axs = plt.subplots(dpi=300)
# colorbarpsf = imshow_field(psf_bb,cmap=color,vmin=-5)
# fig.colorbar(colorbarpsf,ax=axs,orientation='vertical',fraction=0.046,pad=0.04)

# plt.subplot(1,2,1)
# imshow_field(psf_mono,cmap=color,vmin=-5)
# plt.title('Monochromatic PSF')
plt.subplot(1,2,1)
imshow_field(psf_bb,cmap=color,vmin=-4)
#plt.colorbar()
plt.title('Broadband PSF')
plt.subplot(1,2,2)
imshow_field(psf_disp,cmap=color,vmin=-4)
#plt.colorbar()
plt.title('Dispersed Broadband PSF')

plt.tight_layout()
#fig.savefig('/Users/Katie/Documents/MagAO-X/figures/ADC/jatis paper/test.png',dpi=300)
plt.show()