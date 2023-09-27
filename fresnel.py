from poly import *

class phasefilter:

    '''
    This class computes 2D Fourier transform
    of quadratic phase (Fresnel approximation)
    '''
    def __init__(self, petal, embed_factor = 4, freq_ratio = 2):

        if (not isinstance(petal,petal_FT)):
            print('argument must be an instance of the petalFT class')
            return None
        if (petal.n_pixels is None):
            print('n_pixels must be defined at this stage')
            return None

        self.p = petal
        self.n = petal.n_pixels
        self.N = petal.n_pixels * embed_factor
        self.m = self.n // freq_ratio
        self.L = self.N * petal.step / 2.0
        self.phase_step = 2.0*self.L/self.m
        self.phase_axis = np.fft.fftshift(np.fft.fftfreq(self.m,d=1./(2.*self.L)))
        self.Z = float(self.p.occ['Z'])
        self.lambdaRange = self.p.occ['lambdaRange'].squeeze()
        return

    def __call__(self, return_coords=False):

        X,Y = np.meshgrid(self.phase_axis,self.phase_axis)
        filt = np.zeros((X.shape[0],X.shape[1],self.lambdaRange.size),dtype='complex128')
        for i in range(self.lambdaRange.size):
            filt[:,:,i] = np.fft.fft2(np.fft.ifftshift(phase))/self.phase_step**2
        if (return_coords):
            return (X,Y,filt)
        else:
            return (filt)

