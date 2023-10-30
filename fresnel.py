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
        self.L = petal.L
        self.phase_step = 2.0*self.L/self.m
        self.phase_axis = np.fft.fftshift(np.fft.fftfreq(self.m,d=1./(2.*self.L)))
        self.Z = float(self.p.occ['Z'])
        self.lambdaRange = self.p.occ['lambdaRange'].squeeze()
        return

    def __call__(self, return_coords=False):

        if (cuda_on):
            xp = cp
            phase_axis = cp.asarray(self.phase_axis)
        else:
            xp = np
            phase_axis = self.phase_axis
        X,Y = xp.meshgrid(phase_axis,phase_axis)
        phase = xp.zeros_like(X,dtype='complex128')
        filt = np.zeros((X.shape[0],X.shape[1],self.lambdaRange.size),dtype='complex128')
        for i in range(self.lambdaRange.size):
            phase = -1j / (self.lambdaRange[i]*self.Z)*xp.exp(1j*xp.pi*(X**2+Y**2)/(self.lambdaRange[i]*self.Z))
            tmp = xp.fft.ifftshift(phase)
            tmp = xp.fft.fft2(tmp)
            tmp = xp.fft.fftshift(tmp)
            if (cuda_on):
                filt[:,:,i] = cp.asnumpy(tmp)
            else:
                filt[:,:,i] = tmp
        if (cuda_on and return_coords):
            X = cp.asnumpy(X)
            Y = cp.asnumpy(Y)
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
        self.W = compute_W_array(m, step=2.*self.petal.L/self.m)
        return

    def compute_polygonal_fmask(self):
        self.polygonal_fmask = self.petal(self.W).reshape((self.m,self.m))
        return
    
    # def compute_discrete_fmask(self):

    # def compute_discretized_fmask(self, npixels):
        ## TO BE WRITTEN. MAKE SURE TO EXTRACT m*m central frequencies at the end
        ## from the discrete mask Fourier transform
        ## Call poly.pixelized_FT() 

    def compute_fresnel_filter(self):
        self.fresnel_filter = self.phase_filter()
        return

    def compute_diffraction_patterns(self):

        self.diffracted = np.zeros((self.m,self.m,self.phase_filter.lambdaRange.size),dtype='complex128')
        if (cuda_on):
            print ('Cuda is on !')
            xp = cp
            polygonal_fmask = cp.asarray(self.polygonal_fmask)
            diffracted = cp.zeros((self.m,self.m),dtype='complex128')
        else:
            xp = np
            polygonal_fmask = self.polygonal_fmask
            diffracted = np.zeros((self.m,self.m),dtype='complex128')
        for i in range(self.phase_filter.lambdaRange.size):
            if (cuda_on):
                fresnel_filter = cp.asarray(self.fresnel_filter[:,:,i])
            else:
                fresnel_filter = self.fresnel_filter[:,:,i]
            diffracted = 1.0 - xp.fft.fftshift(
                                                xp.fft.ifft2(
                                                    xp.fft.ifftshift(
                                                        fresnel_filter*polygonal_fmask
                                                    )
                                                )
                                              )
            if (cuda_on):
                self.diffracted[:,:,i] = cp.asnumpy(diffracted)
            else:
                self.diffracted[:,:,i] = diffracted
        return

    def compute_extent(self, upper=True):

        L = self.petal.L
        pixel_size = 2.*L / self.m
        if upper:
            extent = (-L-pixel_size/2., L-pixel_size/2.,L-pixel_size/2., -L-pixel_size/2. )
        else:
            extent = (-L-pixel_size/2., L-pixel_size/2.,-L-pixel_size/2., L-pixel_size/2. )
        return (extent)



