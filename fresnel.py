from polygon_fourier import *
# from poly import *

class phasefilter:

    '''
    This class computes 2D Fourier transform
    of quadratic phase (Fresnel approximation)
    '''
    def __init__(self, petal, m=None, Z=None, lambdaRange=None):

        if (not isinstance(petal,(petal_FT,square_FT))):
            print('argument must be an instance of the petalFT class')
            return None
 
        self.p = petal
        if m is None:
            raise ValueError('m must be provided for phasefilter')
        self.m = m
        self.L = petal.L
        self.phase_step = 2.0*self.L/self.m
        self.phase_axis = np.fft.fftshift(np.fft.fftfreq(self.m,d=1./(2.*self.L)))
        if Z is None:
            raise ValueError('Z must be provided for phasefilter')
        self.Z = Z
        if lambdaRange is None:
            raise ValueError('lambdaRange must be provided for phasefilter')
        self.lambdaRange = lambdaRange
        return

    def __call__(self,analytical=False):

        if (cuda_on):
            xp = cp
            phase_axis = cp.asarray(self.phase_axis)
        else:
            xp = np
            phase_axis = self.phase_axis
        if (not analytical):
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
        else: # Use continuous, analytical Fourier tranform of Fresnel phase
            f = xp.fft.fftshift(xp.fft.fftfreq(self.m,d=self.phase_step))
            fxx,fyy = xp.meshgrid(f,f)
            # Use analytical formula
            filt = np.zeros((self.m,self.m,self.lambdaRange.size),dtype='complex128')
            for i in range(self.lambdaRange.size):
                phase = xp.exp(1j*xp.pi*self.lambdaRange[i]*self.Z*(fxx**2+fyy**2))
                if (cuda_on):
                    filt[:,:,i] = cp.asnumpy(phase)
                else:
                    filt[:,:,i] = phase
        return (filt)


class diffraction:

    '''
    This class will call the mask and Fresnel filter related classes, and compute the diffraction figure at
    all wavelengths
    '''

    def __init__(self,petal,m=2**13, embed_factor=4, margin=0.01, Z=None, lambdaRange=None):

        if (not isinstance(petal,(petal_FT, square_FT))):
            print('argument must be an instance of the petalFT class')
            return None
        
        self.m = m
        self.embed_factor=embed_factor
        self.margin=margin
        self.petal = petal
        self.petal.set_the_scene(self.embed_factor,self.margin)
        self.W = compute_W_array(m, step=2.*self.petal.L/self.m)
        if (Z is None and isinstance(self.petal,petal_FT)):
            self.Z = float(self.petal.occ['Z'])
        elif (Z is not None):
            self.Z = Z
        else:
            raise ValueError('Z must be provided for non-petal_FT instances')
        
        if (lambdaRange is None and isinstance(self.petal,petal_FT)):
            self.lambdaRange = self.petal.occ['lambdaRange'][0,:]
        elif (lambdaRange is not None):
            self.lambdaRange = lambdaRange
        else:
            raise ValueError('lambdaRange must be provided for non-petal_FT instances')

        self.phase_filter = phasefilter(self.petal,self.m, self.Z, self.lambdaRange)

        return

    def compute_polygonal_fmask(self,**kwargs):
        self.polygonal_fmask = self.petal(self.W,**kwargs).reshape((self.m,self.m))
        return
    
    # def compute_discrete_fmask(self):

    # def compute_discretized_fmask(self, npixels):
        ## TO BE WRITTEN. MAKE SURE TO EXTRACT m*m central frequencies at the end
        ## from the discrete mask Fourier transform
        ## Call poly.pixelized_FT() 

    def compute_fresnel_filter(self,analytical=False):
        self.fresnel_filter = self.phase_filter(analytical=analytical)
        return

    def compute_diffraction_patterns(self,fmask=None):

        self.diffracted = np.zeros((self.m,self.m,self.phase_filter.lambdaRange.size),dtype='complex128')
        if (fmask is not None):
            fmask = fmask
        else:
            fmask = self.polygonal_fmask

        if (cuda_on):
            print ('Cuda is on !')
            xp = cp
            fmask = cp.asarray(fmask)
            diffracted = cp.zeros((self.m,self.m),dtype='complex128')
        else:
            xp = np
            fmask = fmask
            diffracted = np.zeros((self.m,self.m),dtype='complex128')
        for i in range(self.phase_filter.lambdaRange.size):
            if (cuda_on):
                fresnel_filter = cp.asarray(self.fresnel_filter[:,:,i])
            else:
                fresnel_filter = self.fresnel_filter[:,:,i]
            diffracted = 1.0 - xp.fft.fftshift(
                                                xp.fft.ifft2(
                                                    xp.fft.ifftshift(
                                                        fresnel_filter*fmask
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



