import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from hcipy import *
from psfGenerator import PsfGenerator
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname('adc_sims'), '..')))
from adcFitter_onsky import AdcFitter

import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'x-large',
#          'axes.labelsize': 'x-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'x-large',
#          'ytick.labelsize':'x-large',
#          'font.family':'sans-serif'}
# pylab.rcParams.update(params)

# fieldcube = read_field('/Users/Katie/Documents/Fall 2023/MagAO-X/ADC Sim code/updated onsky/all_lab_fields_i.fits')
# img = fieldcube[0]
# imshow_field(img)

wavelength = 764E-9
bw = 120E-9
wavelength_normalized = wavelength / 656E-9
bw_normalized = bw / wavelength
grating_angle = 28
grating_freq = 40

ADC = AdcFitter(wavelength=wavelength,bandwidth=bw,grating_angle=grating_angle,grating_freq=grating_freq)

PsfGen = PsfGenerator(center_wavelength=wavelength_normalized,bandwidth=bw_normalized, focal_sampling=6, focal_extent=60,grating_angle=grating_angle, fx=grating_freq, fy = grating_freq, mask=True)

def single_speckle_visual(image,speckle_number):
    speckle_angles = np.zeros(4)
    fit_image = image - image
    corners = []

    for i in range(4):
        img = ADC.find_speckle(ADC.psf,i)
        ADC.current_speckle = i
        ADC.set_measurement(img)

        sigma_x = .5#sig_x[i]
        sigma_y = .5#sig_y[i]
        orientation = np.radians(30)

        amplitude = ADC.data.max()
        centroid = ADC.estimate_centroid()
        mu_x = centroid[0]
        mu_y = centroid[1]
        background = 0

        theta_est = np.array([amplitude, mu_x, mu_y, sigma_x, sigma_y, orientation, background])
        fit = ADC.fit(theta_est) 
        theta = fit.x
        corners.append([theta[1],theta[2]])
        speckle_angles[i] = np.degrees(theta[5])
        fit_image = fit_image + ADC.satellite_spot(*theta)(image.grid)
        #print(f'initial orientation value {orientation} fitted value {theta[5]}')
        #print(f'aspect ratio:{np.abs(theta[3]/theta[4])}')

    ADC.current_speckle = None
    speckle_angles = np.array(speckle_angles).T
    print(corners)

    #cutting down to view a single speckle
    size = 120
    #corners = [[20,40],[0,0],[0,0],[0,0]]
    fit_image = ADC.window_field(fit_image,center=corners[speckle_number],width=size,height=size)
    initial_image = ADC.window_field(image,center=corners[speckle_number],width=size,height=size)
    difference = initial_image - fit_image

    return initial_image, fit_image, difference

psf = PsfGen.noisy_turbulent_psf(magnitude_x=0,magnitude_y=0.05,num_photons=1E6,ptv=0,diam=1)
ADC.set_psf(psf)

simulated, fitted_speckle, difference = single_speckle_visual(ADC.psf,speckle_number=2)

plt.figure(dpi=200)
plt.subplot(1,3,1)
imshow_field(simulated,cmap='viridis')
plt.title('Diffraction Speckle')
plt.colorbar()
plt.subplot(1,3,2)
imshow_field(fitted_speckle,cmap='viridis')
plt.title('Fitted 2D Gaussian')
plt.colorbar()
plt.subplot(1,3,3)
imshow_field(difference,cmap='RdBu',vmin=-0.3,vmax=0.3)
plt.colorbar()
plt.title('Difference')
#plt.tight_layout()
plt.show()