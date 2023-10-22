from poly import *

class phasefilter:

    '''
    This class computes 2D Fourier transform
    of quadratic phase (Fresnel approximation)
    '''
    def __init__(self, petal, m=2**13):

        if (not isinstance(petal,petal_FT)):
            print('argument must be an instance of the petalFT class')
            return None
 
        self.p = petal
        self.m = m
        self.L = petal.r_max # Notations...
        self.phase_step = 2.0*self.L/self.m
        self.phase_axis = np.fft.fftshift(np.fft.fftfreq(self.m,d=1./(2.*self.L)))
        self.Z = float(self.p.occ['Z'])
        self.lambdaRange = self.p.occ['lambdaRange'].squeeze()
        return

    def __call__(self, return_coords=False):

        X,Y = np.meshgrid(self.phase_axis,self.phase_axis)
        filt = np.zeros((X.shape[0],X.shape[1],self.lambdaRange.size),dtype='complex128')
        for i in range(self.lambdaRange.size):
            phase = -1j / (self.lambdaRange[i]*self.Z)*np.exp(1j*np.pi*(X**2+Y**2)/(self.lambdaRange[i]*self.Z))
            filt[:,:,i] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phase)))
        if (return_coords):
            return (X,Y,filt)
        else:
            return (filt)


class diffraction:

    '''
    This class will call the mask and Fresnel filter related classes, and compute the diffraction figure at
    all wavelengths
    '''

    def __init__(self,petal,m=2**13, embed_factor=4, margin=0.01):

        self.m = m
        self.embed_factor=embed_factor
        self.margin=margin
        self.petal = petal
        self.petal.set_the_scene(self.embed_factor,self.margin)
        self.phase_filter = phasefilter(self.petal,m=self.m)
        self.W = compute_W_array(m, step=2.*self.petal.r_max/self.m)
        return

    def compute_polygonal_fmask(self):
        self.polygonal_fmask = self.petal(W)
        return

    def compute_discretized_fmask(self, npixels):
        ## TO BE WRITTEN. MAKE SURE TO EXTRACT m*m central frequencies at the end
        ## from the discrete mask Fourier transform
        ## Call poly.pixelized_FT() 

    def compute_fresnel_filter(self):
        self.fresnel_filter = self.phase_filter()
        return

    def compute_diffraction_patterns(self):

        self.diffracted = np.zeros((m,m,self.phase_filter.lambdaRange.size))
        for i in range(self.phase_filter.lambdaRange.size):
            self.diffracted[:,:,i] = 1.0 - np.fft.fftshift(
                                                           np.fft.ifft2(
                                                             np.fft.ifftshift(
                                                               self.fresnel_filter[:,:,i]*self.polygonal_fmask
                                                             )
                                                            )
                                                           )
        return



