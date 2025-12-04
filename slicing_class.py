import numpy as np
from hcipy import *
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import ndimage


class adcFitter2:
    def __init__(self,wavelength=656E-9,bandwidth=100E-9,grating_angle=28,grating_freq=47,ncpc = False,snr_threshold=2,speckle_window=50):
            self.wavelength = wavelength
            self.bandwidth = bandwidth
            self.grating_angle = grating_angle
            self.grating_freq = grating_freq
            self.ncpc = ncpc 
            self.snr_threshold = snr_threshold
            self.normalized_wavelength = wavelength / 656E-9
            self.normalized_bandwidth = bandwidth / 656E-9
            self.control_matrix = np.array([1,1])
            self.speckle_window = speckle_window
            self.hp_sigma = 10
            self.lp_sigma = 5

    def gauss(self,x,mu,sigma2):
        '''standard gaussian function'''
        return np.exp(-(x-mu)**2 / (2 * sigma2) )

    def isolate_gaussian(self,data):
        '''crop off the noisy tails of the gaussian'''
        new_data = data.copy()

        mid = np.argmax(new_data)
        first_derivative = np.gradient(new_data) 

        valley_indices = np.where((first_derivative[:-1] < 0) & (first_derivative[1:] > 0))[0] + 1

        left_valley = valley_indices[valley_indices < mid][-1] if any(valley_indices < mid) else None
        right_valley = valley_indices[valley_indices > mid][0] if any(valley_indices > mid) else None

        if left_valley is not None:
            new_data[0:left_valley] = 0

        if right_valley is not None:
            new_data[right_valley:len(new_data)] = 0

        return new_data

    def mu(self,data):
        '''crop off the noisy tails and find the mean of a single gaussian slice'''
        isolated = self.isolate_gaussian(data)
        x = np.arange(len(data))
        popt = curve_fit(self.gauss,x,isolated,p0=[np.argmax(data),2])

        mu = popt[0][0]
        sigma = popt[0][1]
        fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(sigma)
        return mu, sigma, fwhm
    
    def window_field(self,data,center,width,height):
        '''crop a smaller field out of a big field. data must be an hcipy field.'''
        indx = data.grid.closest_to(center)
        y_ind, x_ind = np.unravel_index(indx, data.shaped.shape)
        cutout = data.shaped[(y_ind-height//2):(y_ind + height//2), (x_ind-width//2):(x_ind+width//2)]
        sub_grid = make_pupil_grid([width, height], [width * data.grid.delta[0], height * data.grid.delta[1]])
        return Field(cutout.ravel(), sub_grid)
        
    def hpf(self,data,sigma):
        return Field((data.shaped - ndimage.gaussian_filter(data.shaped,sigma)).ravel(),data.grid)

    def slice_speckle_angle(self,img,speckle_number,positions=False,print_updates=False,speckle_filter=False):
        '''calculate the angle of a single speckle using the gaussian slicing method. image must be an hcipy field.
        returns the slope of the designated speckle in degrees.
        if positions is set to True, the output will be both the speckle center position and its angle.'''
    
        #locate the general area of the speckle
        extent=20
        ncpc_freq = 29

        if self.ncpc == False: #if we're not using the ncpc speckles, proceed as normal
            speckle_coords = np.array([[0, self.grating_freq * self.normalized_wavelength],[self.grating_freq * self.normalized_wavelength,0],[0, -self.grating_freq * self.normalized_wavelength],[-self.grating_freq * self.normalized_wavelength,0]])
            speckle_center = speckle_coords[speckle_number]
            rect = make_rotated_aperture(make_rectangular_aperture(size=(extent,extent), center=speckle_center), np.deg2rad(-self.grating_angle))(img.grid)
        else: #if we are, there's a different rotation angle for speckles 1&3 and 2&4
            speckle_coords = np.array([[ncpc_freq * self.normalized_wavelength,0],[self.grating_freq * self.normalized_wavelength,0],[-ncpc_freq * self.normalized_wavelength,0],[-self.grating_freq * self.normalized_wavelength,0]])
            speckle_center = speckle_coords[speckle_number]
            if speckle_number==0 or speckle_number==2:
                rect = make_rectangular_aperture(size=(extent,extent), center=speckle_center)(img.grid)
            else:
                rect = make_rotated_aperture(make_rectangular_aperture(size=(extent,extent), center=speckle_center), np.deg2rad(-self.grating_angle))(img.grid)

        speckle_img = rect * img

        #find the center of intensity of the speckle area and window the ORIGINAL image around that center
        #center_of_intensity = np.array([sum(speckle_img*speckle_img.grid.x)/sum(speckle_img),sum(speckle_img*speckle_img.grid.y)/sum(speckle_img)])
        max_pixel = speckle_img.grid[np.argmax(speckle_img)]
        #print(f'coord of max pixel: {max_pixel}')
        window_size=self.speckle_window
        #new_img = self.window_field(img,[center_of_intensity[0],center_of_intensity[1]],window_size,window_size)
        new_img = self.window_field(img,[max_pixel[0],max_pixel[1]],window_size,window_size)

        #filter the speckles
        if speckle_filter:
            hp = self.hpf(new_img,self.hp_sigma)
            lp = hp - self.hpf(hp,self.lp_sigma)
            lp -= np.median(lp)
            lp[lp < 0 ] = 0
            new_img = lp
            new_img = new_img / np.max(new_img)

        shaped = new_img.shaped
        mus = np.zeros(shaped.shape[0])

        for i in range(shaped.shape[0]):
            if self.ncpc==True:
                sliced = shaped[:,i] 
            elif speckle_number % 2 != 0: #columns vs rows depending on which speckle it is
                sliced = shaped[:,i] 
            else: 
                sliced = shaped[i,:]

        # for i in range(shaped.shape[0]): 
        #     if speckle_number % 2 != 0: #columns vs rows depending on which speckle it is
        #         sliced = shaped[:,i]
        #     else: sliced = shaped[i,:]

            zscore = (np.max(sliced) - np.mean(sliced))/np.std(sliced)

            if zscore > self.snr_threshold:
                s = self.isolate_gaussian(sliced)
                s /= s.max()
                try:
                    m, std, fwhm = self.mu(s)
                except:
                    print(f'fitting error. could not fit speckle {speckle_number}')
                    return np.nan
                
                mus[i] = m
            else: mus[i] = -1

        #take the second derivative so we can isolate the nonlinear region
        deriv2 = np.gradient(np.gradient(mus))
        nonlin_region = np.atleast_1d(np.squeeze(np.where(np.abs(deriv2) >= 0.4)))

        if len(nonlin_region)==0:
            print('edges of speckle not detected. either speckle takes up entire window, or snr threshold needs to be changed')

        #start from the middle of the mu vector and move outward to find the bounds of the linear region
        midpoint = int(np.floor(window_size / 2))

        if any(nonlin_region < midpoint):
            leftbound = nonlin_region[nonlin_region < midpoint][-1]
        elif deriv2[0] <= 0.4:
            leftbound = 0
            #print('speckle falls off left (bottom) edge')
        else: leftbound = None

        if any(nonlin_region > midpoint):
            rightbound = nonlin_region[nonlin_region > midpoint][0]
        elif deriv2[-1] <= 0.4:
            rightbound = 50
            #print('speckle falls off right (top) edge')
        else: rightbound = None

        # leftbound = nonlin_region[nonlin_region < midpoint][-1] if any(nonlin_region < midpoint) else None
        # rightbound = nonlin_region[nonlin_region > midpoint][0] if any(nonlin_region > midpoint) else None
        # print(f'leftbound: {leftbound}')
        # print(f'rightbound: {rightbound}')

        if leftbound != None and rightbound != None:
            lin_region_center = (leftbound+rightbound)//2 #this is the center WRT the new window. to do this properly you need it in terms of the old window.
            corresponding_mu = mus[lin_region_center]
            relative_center = np.array([lin_region_center - self.speckle_window//2,corresponding_mu - self.speckle_window//2])
            #print(f'relative center {relative_center}')
            if speckle_number % 2 != 0: #columns vs rows depending on which speckle it is
                #centroid = np.array([center_of_intensity[0] + relative_center[0]*speckle_img.grid.delta[0],center_of_intensity[1] + relative_center[1] * speckle_img.grid.delta[1]])
                centroid = np.array([max_pixel[0] + relative_center[0]*speckle_img.grid.delta[0],max_pixel[1] + relative_center[1] * speckle_img.grid.delta[1]])
                #print(centroid)
            else:
                relative_center = np.flip(relative_center) 
                #centroid = np.array([center_of_intensity[0] + relative_center[0]*speckle_img.grid.delta[0],center_of_intensity[1] + relative_center[1] * speckle_img.grid.delta[1]])
                centroid = np.array([max_pixel[0] + relative_center[0]*speckle_img.grid.delta[0],max_pixel[1] + relative_center[1] * speckle_img.grid.delta[1]])
                #print(centroid)

        #do a linear regression on the mus in the linear region. if there is no clear linear region, print an error if print_updates is enabled.
        if leftbound is not None and rightbound is not None:
            linear_region = mus[leftbound:rightbound]
            lin_x = np.arange(len(linear_region))

            m , b = np.polyfit(lin_x,linear_region,deg=1)
            angle = np.arctan(m)
            # if speckle_number % 2 == 0:
            #     angle = np.pi /2 - angle

            if print_updates ==True:
                print(f'slope: {m:.2f}\ncorresponding angle: {angle:.2f} (rad) or {np.degrees(angle):.2f}°')

            if positions==False:
                return np.degrees(angle) #returns the slope of the individual speckle in degrees.
            else: 
                return np.degrees(angle),centroid #returns both the slope and the speckle's center position.
        else: 
            if print_updates == True:
                print(f'unable to fit speckle {speckle_number}')
            return np.nan

    def all_speckle_angles(self,img,speckle_filter=False):
        '''returns a numpy vector containing the four speckle angles found in the image.
        need a way to incorporate the snr threshold??
        '''
        angles = np.zeros(4)
        for i in range(4):
            #print(i)
            angles[i] = self.slice_speckle_angle(img,i,speckle_filter=speckle_filter)
        return angles
    
    def crop_image(self, image,extent=400,mask_diam=60): 
        '''cuts out a centered PSF with the central core masked'''
        img = image/image.max()

        img_subtracted = img >0.1
        center_of_intensity = np.array([sum(img_subtracted*img_subtracted.grid.x)/sum(img_subtracted),sum(img_subtracted*img_subtracted.grid.y)/sum(img_subtracted)])
        mask_ap = make_circular_aperture(mask_diam,center_of_intensity)
        mask = mask_ap(img_subtracted.grid)
        mask = abs(mask - 1)
        masked_img = mask * img

        img = masked_img
        img = self.window_field(img,[center_of_intensity[0],center_of_intensity[1]],extent,extent)
        img /= img.max()
        
        # bk = np.median(img)
        # img -= bk
        
        mask2 = make_circular_aperture(mask_diam,np.array([0,0]))(img.grid)
        mask2 = abs(mask2-1)
        img = mask2 * img
        img = Field([x if x>0 else 0 for x in img],img.grid)

        return img
    
    def filter_image(self,img,low_freq = 0.01,high_freq=1):

        ff = FourierFilter(img.grid, make_circular_aperture(2 * np.pi * low_freq))
        filtered_img= np.real(ff.forward(img + 0j))
        img = img - filtered_img

        ff2 = FourierFilter(img.grid, make_circular_aperture(2 * np.pi * high_freq))
        filtered_img = np.real(ff2.forward(img + 0j))
        img = filtered_img

        # binc, profile, std_profile, ncount = radial_profile(img,.25)
        # r_coordinates = img.grid.as_('polar').r
        # radial_map = np.interp(r_coordinates, binc, profile)

        filtered_subtracted = img #- radial_map
        filtered_subtracted = Field([x if x>0 else 0 for x in img],filtered_subtracted.grid)
        
        return filtered_subtracted
    
    def speckle_pairs(self,a):
        diff13 = a[0] - a[2]
        diff24 = a[1] - a[3]
        pairs = np.array([diff13,diff24])
        return pairs

    def calculate_command(self,speckle_pairs):
        predicted_disp = self.control_matrix @ speckle_pairs.T
        predicted_disp = np.array(predicted_disp)
        return -predicted_disp