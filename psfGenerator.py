import numpy as np
from hcipy import *    

class PsfGenerator:
    def __init__(self, telescope_pupil=None, center_wavelength=1, bandwidth = 0.1, sparkles=True,fx = 10, fy = 10, focal_sampling = 3, focal_extent = 16, ccd_grid=None, mask=False, grating_angle=0,polar=False):
        #creating Magellan pupil as default
        if telescope_pupil is None:
            pupil_grid = make_pupil_grid(128)
            aperture = make_magellan_aperture(normalized=True)
            self.telescope_pupil = evaluate_supersampled(aperture,pupil_grid,4)
        else: 
            self.telescope_pupil = telescope_pupil

        #waveband information
        self.center_wavelength = center_wavelength
        self.bandwidth = bandwidth

        #grating to create DM speckles
        self.grating_angle = np.radians(grating_angle)
        self.grating_fx = fx
        self.grating_fy = fy
        self.grating_optic = self.grating()
        self.sparkles = sparkles
 
        #propagation
        if ccd_grid == None:
            self.focal_sampling = focal_sampling
            self.focal_grid = make_focal_grid(focal_sampling,focal_extent)
        else:
            self.focal_grid = ccd_grid.scaled(668e-9/1) #scaled by lam/D at our halpha filter wavelength, which are the ccd pixel units

        self.propagate = FraunhoferPropagator(self.telescope_pupil.grid,self.focal_grid)

        #masking the airy core
        self.mask = mask

        #coronagraph
        #lyot coronagraph attempt
        # lyot_fpm = evaluate_supersampled(make_circular_aperture(3), self.focal_grid, 4) #a 3 lam/d lyot focal plane mask
        # lyot_fpm = np.abs(lyot_fpm -1)
        # lyot_stop = evaluate_supersampled(make_circular_aperture(0.95), pupil_grid, 4)
        # self.cor = LyotCoronagraph(pupil_grid,lyot_fpm,lyot_stop=None,focal_length=.759,focal_plane_mask_grid=lyot_fpm.grid) #not sure about the focal length here
        
        #using PerfectCoronagraph
        self.cor = PerfectCoronagraph(self.telescope_pupil,2)

    def grating(self):
        ''' creating the speckle patterns '''

        #creating the cosine pattern
        grating_grid = self.telescope_pupil.grid
        rotated_grid = grating_grid.rotated(self.grating_angle)
        cos_pattern_x = Field(np.cos(2*np.pi*self.grating_fx * rotated_grid.x),grating_grid)
        cos_pattern_y = Field(np.cos(2*np.pi*self.grating_fy * rotated_grid.y),grating_grid)

        #applying the cosine pattern to the phase apodizer
        gratings = PhaseApodizer(cos_pattern_x+cos_pattern_y)
        return gratings

    def psf(self, center_x=0, center_y=0, magnitude_x=0, magnitude_y=0):
        '''creates a broadband dispersed PSF with the specified magnitude, orientation, center wavelength, and bandwidth. the center 
        coordinates of the psf are in wavelength units, i.e. 1 = 668E-9'''
        '''currently makes 10 PSFs spaced evenly by wavelength'''
        '''outputs an intensity field only'''

        def disp(lam,lam0): #linear function of wavelength and the center wavelength
            dist = (lam - lam0) #distance from center wavelength
            return -dist * 100
        
        TT = TipTiltMirror(self.telescope_pupil.grid)
        TT.actuators[0] = center_x/2 * self.center_wavelength #center shift in the x direction in units of waves
        TT.actuators[1] = center_y/2 * self.center_wavelength #center shift in the y direction 
        
        bb_psf = 0
                        
        for wlen in np.linspace(self.center_wavelength - self.bandwidth / 2., self.center_wavelength + self.bandwidth / 2., 12):
            k = 2 * np.pi / wlen #wavemumber
            opd_y = magnitude_y * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.y
            opd_x = magnitude_x * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.x
            opd = opd_y + opd_x 
            phase = opd * k #optical path difference between dispersed & not dispersed
            
            wf1 = Wavefront(self.telescope_pupil * np.exp(1j*phase),wavelength=wlen)
            wf1.total_power = 1/12 #normalizing each wavefront so that the total number of photons is 1 in the BB PSF
            
            if self.sparkles:
                wf2 = self.grating_optic.forward(wf1)
            else:
                wf2 = wf1
            
            wf3 = TT.forward(wf2)
            wf3.total_power = 1/12
            wf4 = self.propagate.forward(wf3)
            power = wf4.power
            
            if self.mask == True:
                mask_ap = make_circular_aperture(self.center_wavelength*10,[center_x * self.center_wavelength,center_y * self.center_wavelength])
                masking = mask_ap(power.grid)
                masking = np.abs(masking - 1)
                power *= masking
            
            bb_psf += power

        return bb_psf
    

    def noisy_psf(self, center_x=0, center_y=0, magnitude_x=0, magnitude_y=0, num_photons=1E8):
        '''creates a broadband dispersed PSF with the specified magnitude, orientation, center wavelength, and bandwidth. the center 
        coordinates of the psf are in wavelength units, i.e. 1 = 668E-9'''
        '''currently makes 12 PSFs spaced evenly by wavelength'''
        '''outputs an intensity field only'''

        def disp(lam,lam0): #linear function of wavelength and the center wavelength
            dist = (lam - lam0) #distance from center wavelength
            return -dist * 100
        
        TT = TipTiltMirror(self.telescope_pupil.grid)
        TT.actuators[0] = center_x/2 * self.center_wavelength #center shift in the x direction in units of waves
        TT.actuators[1] = center_y/2 * self.center_wavelength #center shift in the y direction 
        
        bb_psf = 0
                        
        for wlen in np.linspace(self.center_wavelength - self.bandwidth / 2., self.center_wavelength + self.bandwidth / 2., 12):
            k = 2 * np.pi / wlen #wavemumber
            opd_y = magnitude_y * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.y
            opd_x = magnitude_x * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.x
            opd = opd_y + opd_x 
            phase = opd * k #optical path difference between dispersed & not dispersed
            
            wf1 = Wavefront(self.telescope_pupil * np.exp(1j*phase),wavelength=wlen)
            wf1.total_power = 1/15 #normalizing each wavefront so that the total number of photons is 1 in the BB PSF
            if self.sparkles:
                wf2 = self.grating_optic.forward(wf1)
            else:
                wf2 = wf1
            wf3 = TT.forward(wf2)
            wf4 = self.propagate.forward(wf3)
            power = wf4.power
            
            if self.mask == True:
                mask_ap = make_circular_aperture(self.center_wavelength*10,[center_x * self.center_wavelength,center_y * self.center_wavelength])
                masking = mask_ap(power.grid)
                masking = np.abs(masking - 1)
                power *= masking
            
            bb_psf += power

        bb_psf = bb_psf / sum(bb_psf)

        #adding in poisson noise
        bb_psf = large_poisson(num_photons * bb_psf)
        #bb_psf /= bb_psf.max()

        return bb_psf
    
    def noisy_turbulent_psf(self,center_x=0,center_y=0,magnitude_x=0,magnitude_y=0,num_photons=1E6,ptv=0.2,diam=1,exponent=-2.5):
        '''a PSF generator that includes photon noise and atmospheric turbulence simulated by a power-law distribution'''
        def disp(lam,lam0): #linear function of wavelength and the center wavelength
            dist = (lam - lam0) #distance from center wavelength
            #return magnitude * dist * 100
            return -dist * 100
        
        TT = TipTiltMirror(self.telescope_pupil.grid)
        TT.actuators[0] = center_x/2 * self.center_wavelength #center shift in the x direction in units of waves
        TT.actuators[1] = center_y/2 * self.center_wavelength #center shift in the y direction 
        
        abr = SurfaceAberration(self.telescope_pupil.grid,ptv=ptv,diameter=diam,exponent=exponent)

        bb_psf = 0

        #generating the phase turbulence
        # phase_aberrated = make_power_law_error(self.telescope_pupil.grid,ptv=ptv,diameter=diam,exponent=exponent)
        # phase_aberrated -= np.mean(phase_aberrated[self.telescope_pupil >= 0.5])
                        
        for wlen in np.linspace(self.center_wavelength - self.bandwidth / 2., self.center_wavelength + self.bandwidth / 2., 12):
            k = 2 * np.pi / wlen #wavenumber
            opd_y = magnitude_y * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.y
            opd_x = magnitude_x * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.x
            opd = opd_y + opd_x 
            phase = opd * k #optical path difference between dispersed & not dispersed

            #wf1 = Wavefront(self.telescope_pupil * np.exp(1j*(phase + phase_aberrated)),wavelength=wlen)
            wf1 = Wavefront(self.telescope_pupil * np.exp(1j*(phase)),wavelength=wlen)
            wf1.total_power = 1/12 #normalizing each wavefront so that the total number of photons is 1 in the BB PSF
            wfabr = abr.forward(wf1)
            if self.sparkles:
                wf2 = self.grating_optic.forward(wfabr)
            else:
                wf2 = wfabr
            wf3 = TT.forward(wf2)
            wf4 = self.propagate.forward(wf3)
            power = wf4.power
            
            if self.mask == True:
                mask_ap = make_circular_aperture(self.center_wavelength*10,[center_x * self.center_wavelength,center_y * self.center_wavelength])
                masking = mask_ap(power.grid)
                masking = np.abs(masking - 1)
                power *= masking
            
            bb_psf += power

        #adding in poisson noise
        bb_psf = large_poisson(num_photons * bb_psf)
        #bb_psf /= bb_psf.max()

        return bb_psf
    
    def polar_psf(self,center_x=0,center_y=0,magnitude=0,orientation=0,num_photons=1E6,ptv=0.2,diam=1,exponent=-2.5):
        '''PSF generator that generates dispersion in terms of magnitude and orientation, like the MagAO-X ADCs do. Orientation should be entered in degrees, and is measured from the positive x-axis.'''
        def disp(lam,lam0): #linear function of wavelength and the center wavelength
            dist = (lam - lam0) #distance from center wavelength
            #return magnitude * dist * 100
            return -dist * 100
        
        TT = TipTiltMirror(self.telescope_pupil.grid)
        TT.actuators[0] = center_x/2 * self.center_wavelength #center shift in the x direction in units of waves
        TT.actuators[1] = center_y/2 * self.center_wavelength #center shift in the y direction 
        
        abr = SurfaceAberration(self.telescope_pupil.grid,ptv=ptv,diameter=diam,exponent=exponent)

        bb_psf = 0

        magnitude_x = magnitude * np.cos(-np.radians(orientation))
        magnitude_y = magnitude * np.sin(-np.radians(orientation))

        # I realize I could just do that calculation ^^ and then call the previous function, but whatever.
                        
        for wlen in np.linspace(self.center_wavelength - self.bandwidth / 2., self.center_wavelength + self.bandwidth / 2., 12):
            k = 2 * np.pi / wlen #wavenumber
            opd_y = magnitude_y * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.y
            opd_x = magnitude_x * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.x
            opd = opd_y + opd_x 
            phase = opd * k #optical path difference between dispersed & not dispersed

            wf1 = Wavefront(self.telescope_pupil * np.exp(1j*(phase)),wavelength=wlen)
            wf1.total_power = 1/12 #normalizing each wavefront so that the total number of photons is 1 in the BB PSF
            wfabr = abr.forward(wf1)
            if self.sparkles:
                wf2 = self.grating_optic.forward(wfabr)
            else:
                wf2 = wfabr
            wf3 = TT.forward(wf2)
            wf4 = self.propagate.forward(wf3)
            power = wf4.power
            
            if self.mask == True:
                mask_ap = make_circular_aperture(self.center_wavelength*10,[center_x * self.center_wavelength,center_y * self.center_wavelength])
                masking = mask_ap(power.grid)
                masking = np.abs(masking - 1)
                power *= masking
            
            bb_psf += power

        #adding in poisson noise
        bb_psf = large_poisson(num_photons * bb_psf)
        #bb_psf /= bb_psf.max()

        return bb_psf
    
    def coron_psf(self,center_x=0,center_y=0,magnitude_x=0,magnitude_y=0,num_photons=1E6,ptv=0,diam=1,exponent=-2.5):
        '''a PSF generator that includes photon noise and atmospheric turbulence simulated by a power-law distribution'''
        def disp(lam,lam0):
            dist = (lam - lam0)
            return -dist * 100
        
        TT = TipTiltMirror(self.telescope_pupil.grid)
        TT.actuators[0] = center_x/2 * self.center_wavelength #units of waves
        TT.actuators[1] = center_y/2 * self.center_wavelength 
        
        abr = SurfaceAberration(self.telescope_pupil.grid,ptv=ptv,diameter=diam,exponent=exponent)
        coron = self.cor

        bb_psf = 0
                        
        for wlen in np.linspace(self.center_wavelength - self.bandwidth / 2., self.center_wavelength + self.bandwidth / 2., 12):
            k = 2 * np.pi / wlen #wavenumber
            opd_y = magnitude_y * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.y
            opd_x = magnitude_x * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.x
            opd = opd_y + opd_x 
            phase = opd * k #optical path difference between dispersed & not dispersed

            wf1 = Wavefront(self.telescope_pupil * np.exp(1j*(phase)),wavelength=wlen)
            wf1.total_power = 1/12 
            wfabr = abr.forward(wf1)
            if self.sparkles:
                wf2 = self.grating_optic.forward(wfabr)
            else:
                wf2 = wfabr
            wf3 = TT.forward(wf2)
            wf3.total_power = 1/12
            wf4 = self.propagate.forward(coron(wf3))
            power = wf4.power
            
            if self.mask == True:
                mask_ap = make_circular_aperture(self.center_wavelength*10,[center_x * self.center_wavelength,center_y * self.center_wavelength])
                masking = mask_ap(power.grid)
                masking = np.abs(masking - 1)
                power *= masking
            
            bb_psf += power

        #adding in poisson noise
        #bb_psf = large_poisson(num_photons * bb_psf)
        # if bb_psf.max() != 0:
        #     bb_psf /= bb_psf.max()

        return bb_psf
    

    def lyot_coron_psf(self,fpm,lyot_stop,center_x=0,center_y=0,magnitude_x=0,magnitude_y=0,num_photons=1E6,ptv=0,diam=1,exponent=-2.5):
        '''a PSF generator that includes photon noise and atmospheric turbulence simulated by a power-law distribution'''
        def disp(lam,lam0):
            dist = (lam - lam0)
            return -dist * 100
        
        TT = TipTiltMirror(self.telescope_pupil.grid)
        TT.actuators[0] = center_x/2 * self.center_wavelength #units of waves
        TT.actuators[1] = center_y/2 * self.center_wavelength 
        
        abr = SurfaceAberration(self.telescope_pupil.grid,ptv=ptv,diameter=diam,exponent=exponent)
        coron = LyotCoronagraph(self.telescope_pupil.grid,fpm,lyot_stop,focal_length=1,focal_plane_mask_grid=fpm.grid)

        bb_psf = 0
                        
        for wlen in np.linspace(self.center_wavelength - self.bandwidth / 2., self.center_wavelength + self.bandwidth / 2., 12):
            k = 2 * np.pi / wlen #wavenumber
            opd_y = magnitude_y * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.y
            opd_x = magnitude_x * disp(wlen,self.center_wavelength) * self.telescope_pupil.grid.x
            opd = opd_y + opd_x 
            phase = opd * k #optical path difference between dispersed & not dispersed

            wf1 = Wavefront(self.telescope_pupil * np.exp(1j*(phase)),wavelength=wlen)
            wf1.total_power = 1/12 
            wfabr = abr.forward(wf1)
            if self.sparkles:
                wf2 = self.grating_optic.forward(wfabr)
            else:
                wf2 = wfabr
            wf3 = TT.forward(wf2)
            wf3.total_power = 1/12
            wf4 = self.propagate.forward(coron(wf3))
            power = wf4.power
            
            if self.mask == True:
                mask_ap = make_circular_aperture(self.center_wavelength*10,[center_x * self.center_wavelength,center_y * self.center_wavelength])
                masking = mask_ap(power.grid)
                masking = np.abs(masking - 1)
                power *= masking
            
            bb_psf += power

        #adding in poisson noise
        #bb_psf = large_poisson(num_photons * bb_psf)
        # if bb_psf.max() != 0:
        #     bb_psf /= bb_psf.max()

        return bb_psf