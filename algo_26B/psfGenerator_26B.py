import numpy as np
from hcipy import *   

def disp(lam,lam0): #linear function of wavelength and the center wavelength
    dist = (lam - lam0) #distance from center wavelength
    return -dist #need to multiply by converesion factor so that the dispersion is in mas units

def psf(wavelength,bandwidth,dispersion_x=0, dispersion_y=0,freq=10,angle=0,mask=False):
    telescope_diameter = 6.5 # meter
    oversizing_factor = 1 #how much extra space next to the pupil (1 = no padding)

    #define pupil grid
    num_pupil_pixels = 256
    pupil_grid_diameter = telescope_diameter * oversizing_factor
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

    #create pupil
    magellan_aperture_generator = make_magellan_aperture(False) #diameter is automatically 6.5
    magellan_aperture = evaluate_supersampled(magellan_aperture_generator, pupil_grid, 4)

    #set up wavefront for propagation
    wf = Wavefront(magellan_aperture, wavelength)
    wf.total_power = 1

    #propagator
    spatial_resolution = wavelength / telescope_diameter
    focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

    propagator = FraunhoferPropagator(pupil_grid, focal_grid)

    ###diffraction grating
    #creating the cosine pattern
    grating_grid = magellan_aperture.grid
    angle = np.radians(angle)
    rotated_grid = grating_grid.rotated(angle)
    cos_pattern_x = Field(np.cos(2*np.pi*(freq/telescope_diameter) * rotated_grid.x),grating_grid)
    cos_pattern_y = Field(np.cos(2*np.pi*(freq/telescope_diameter) * rotated_grid.y),grating_grid)

    #applying the cosine pattern to the phase apodizer
    grating = PhaseApodizer(cos_pattern_x+cos_pattern_y)

    #propagation
    wf1 = grating.forward(wf)
    wf2 = propagator.forward(wf1).power

    bb_psf = 0
                    
    for wlen in np.linspace(wavelength - bandwidth / 2., wavelength + bandwidth / 2., 15):
        k = 2 * np.pi / wlen #wavemumber
        opd_y = dispersion_y * disp(wlen,wavelength) * pupil_grid.y
        opd_x = dispersion_x * disp(wlen,wavelength) * pupil_grid.x
        opd = opd_y + opd_x 
        phase = opd * k #optical path difference between dispersed & not dispersed
        
        wf1 = Wavefront(magellan_aperture * np.exp(1j*phase),wavelength=wlen)
        wf1.total_power = 1/15 #normalizing each wavefront so that the total number of photons is 1 in the BB PSF
        wf2 = grating.forward(wf1)
        wf3 = propagator.forward(wf2)
        power = wf3.power
        
        if mask:
            mask_ap = make_circular_aperture(wavelength*10,[0,0])
            masking = mask_ap(power.grid)
            masking = np.abs(masking - 1)
            power *= masking
        
        bb_psf += power

    return bb_psf

def poisson_psf(wavelength,bandwidth,num_photons,dispersion_x=0, dispersion_y=0,freq=10,angle=0,mask=False,background=.01):
    #background parameter is a proportion of the maximum signal, added to everything that is 2 sigma away from the maximum 

    telescope_diameter = 6.5 # meter
    oversizing_factor = 1 #how much extra space next to the pupil (1 = no padding)

    #define pupil grid
    num_pupil_pixels = 256
    pupil_grid_diameter = telescope_diameter * oversizing_factor
    pupil_grid = make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

    #create pupil
    magellan_aperture_generator = make_magellan_aperture(False) #diameter is automatically 6.5
    magellan_aperture = evaluate_supersampled(magellan_aperture_generator, pupil_grid, 4)

    #set up wavefront for propagation
    wf = Wavefront(magellan_aperture, wavelength)
    wf.total_power = 1

    #propagator
    spatial_resolution = wavelength / telescope_diameter
    focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

    propagator = FraunhoferPropagator(pupil_grid, focal_grid)

    ###diffraction grating
    #creating the cosine pattern
    grating_grid = magellan_aperture.grid
    angle = np.radians(angle)
    rotated_grid = grating_grid.rotated(angle)
    cos_pattern_x = Field(np.cos(2*np.pi*(freq/telescope_diameter) * rotated_grid.x),grating_grid)
    cos_pattern_y = Field(np.cos(2*np.pi*(freq/telescope_diameter) * rotated_grid.y),grating_grid)

    #applying the cosine pattern to the phase apodizer
    grating = PhaseApodizer(cos_pattern_x+cos_pattern_y)

    #propagation
    wf1 = grating.forward(wf)
    wf2 = propagator.forward(wf1).power

    bb_psf = 0
                    
    for wlen in np.linspace(wavelength - bandwidth / 2., wavelength + bandwidth / 2., 15):
        k = 2 * np.pi / wlen #wavemumber
        opd_y = dispersion_y * disp(wlen,wavelength) * pupil_grid.y
        opd_x = dispersion_x * disp(wlen,wavelength) * pupil_grid.x
        opd = opd_y + opd_x 
        phase = opd * k #optical path difference between dispersed & not dispersed
        
        wf1 = Wavefront(magellan_aperture * np.exp(1j*phase),wavelength=wlen)
        wf1.total_power = 1/15 #normalizing each wavefront so that the total number of photons is 1 in the BB PSF
        wf2 = grating.forward(wf1)
        wf3 = propagator.forward(wf2)
        power = wf3.power
        
        if mask:
            mask_ap = make_circular_aperture(wavelength*10,[0,0])
            masking = mask_ap(power.grid)
            masking = np.abs(masking - 1)
            power *= masking
        
        bb_psf += power

    bb_psf = large_poisson(num_photons * bb_psf)

    return bb_psf