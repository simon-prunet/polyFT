###
### Diagonal of matrix product, used to compute element wise (2D) dot products on rows and columns, 
### is obtained in the following way: a_{ii} = \sum_j b_{ij} c^T_{ji} = \sum_j b_{ij} c_{ij}
### which in numpy language is written (B*C).sum(-1)

# BEWARE
# All things related to the radial profile use r_last (last defined point in SISTER profile), 
# including contour samples for polygonal transform.
# BUT all pixel/frequency axes use r_out = occulterDiameter/2


try:
    import cupy as cp
    import numpy as np
    cuda_on = True
except:
    import numpy as np
    cuda_on = False

# Essai !!
# cuda_on = False

import scipy as sp
import os
from scipy.io import loadmat

def get_array_module(arr):
    '''
    if cuda is available, and arr is a cupy array, returns cupy, otherwise returns numpy
    '''
    if (cuda_on):
        xp = cp.get_array_module(arr)
    else:
        xp = np
    return(xp)


def rot(arr):
    xp = get_array_module(arr)
    return xp.vstack((-arr[:,1],arr[:,0])).T

class polyFT:
    
    def __init__(self, Gamma, sinc_formula=True, **kwargs):

        '''
        Initializes the polygonal Fourier Transform class.
        This will allow to compute the Fourier Transform of the indicatrix 
        function of an arbitrary polygonal surface, at an arbitrary set of 
        positions in uv space.

        Takes as input the 2D coordinates of the polygone summits.
        '''

        self.Gamma = Gamma
        self.npoints = Gamma.shape[0]
        if sinc_formula is True:
            self.sinc_formula=True
            # Use formula based on sinc, from J. Wuttke (arxiv:1703.00255, math-ph)
            # This formula uses half polygone edge vectors and coordinates of edge middle
            self.Ej = (np.roll(Gamma,-1,axis=0) - Gamma)/2.
            self.Rj = (np.roll(Gamma,-1,axis=0) + Gamma)/2.
        else:
            # Use formula from 1983 original paper 
            # Compute normalized polygone edges \alpha_n
            #self.Alpha = Gamma - np.roll(Gamma,1,axis=0)
            self.sinc_formula=False
            self.Alpha = np.roll(Gamma,-1,axis=0) - Gamma
            self.Alpha /= np.linalg.norm(self.Alpha,axis=1)[:,None]
            # Also keep shifted Alpha matrix handy
            self.Alpha_m1 = np.roll(self.Alpha,1,axis=0)
            self.num_weight = ( rot(self.Alpha)*self.Alpha_m1 ).sum(-1)

        return

    def area (self):
        '''
        Computes polygonal area (zero-frequency term of Fourier transform)
        '''
        Gamma = self.Gamma
        res = 0.5 * np.sum(-np.roll(rot(Gamma),1,axis=0)*Gamma) # \sum [\hat{n},V_{j-1},V_{j}]
        return (res)
    
    def process (self,w):
        
        '''
        Computes Fourier transform of indicatrix function of polygonal shape,
        at 2D positions in Fourier space specified by the W matrix
        '''

        # Beware that W are wave vectors in the paper, but are assumed to be spatial frequencies here,
        # to allow direct comparisons to FFT computations, hence the 2pi factors in denominator and phase.
        # Note that this is purely conventional.
        # If W is a cupy array, computations will be done on GPU and the result will be a cupy array
        
        xp = get_array_module(w)
        Gamma = xp.asarray(self.Gamma)

        if self.sinc_formula is True:
            Rj = xp.asarray(self.Rj)
            Ej = xp.asarray(self.Ej)
            wx = rot(w)

            # print('After allocating Ej', cp._default_memory_pool.used_bytes())

            num_weight = xp.exp(2j*xp.pi*xp.dot(w,Rj.T)) #phase term
            # print('After allocating phase', cp._default_memory_pool.used_bytes())
            num_weight *= xp.sinc(2.*xp.dot(w,Ej.T)) # sinc term
            num_weight *= np.dot(wx,Ej.T) # geometric term
            
            # print('After computing num_weight', cp._default_memory_pool.used_bytes())

            result = -num_weight.sum(-1) / xp.linalg.norm(w,axis=1)**2 / (1j*xp.pi) # 1/q^2 term
            # Take care of W=(0,0) null frequency case: result is polygone area
            result[xp.linalg.norm(w,axis=1)==0] = 0.5 * xp.sum(-xp.roll(rot(Gamma),1,axis=0)*Gamma)
            return (result)
        else: 
            # old formula
            Alpha = xp.asarray(self.Alpha)
            Alpha_m1 = xp.asarray(self.Alpha_m1)
            num_weight = xp.asarray(self.num_weight)

            den_weight = xp.dot(w,Alpha.T) 
            den_weight *= xp.dot(w,Alpha_m1.T) * (2.*xp.pi)**2
            weight = xp.exp(2j*xp.pi*xp.dot(w,Gamma.T))/den_weight
            return (num_weight[None,:]*weight).sum(-1)

    def __call__ (self,W,cpu_memory_limit=50,gpu_memory_limit=10):

        '''
        Call the process function in a loop to avoid memory overload, 
        especially when computing on GPU.
        cpu and gpu memory limits are expressed in GigaBytes.
        '''

        cpu_limit = cpu_memory_limit * 1024**3
        gpu_limit = gpu_memory_limit * 1024**3
        # Memory allocation will be dominated by the different dot (tensor) products
        # There are three of them of size W.shape[0]*Gamma.shape[0]*sizeof(complex128)
        nw = W.shape[0]
        npo = self.npoints
        if (cuda_on):
            nslices = 3* nw*npo*16 // gpu_limit +1 # Complex numbers, double precision
        else:
            nslices = 3* nw*npo*16 // cpu_limit +1

        print ('nslices = ',nslices)

        res = np.zeros(nw,dtype=np.complex128)
        indices = np.array_split(np.arange(nw),nslices)
        for i in range(nslices):
            print ('Processing slice number %d out of %d'%(i,nslices))
            if (cuda_on):
                wi = cp.asarray(W[indices[i],:])
                print ('shape of wi is ',wi.shape)
                print(cp._default_memory_pool.used_bytes())
                resi = self.process(wi)
                res[indices[i]] = cp.asnumpy(resi)
                del wi, resi # Clean GPU memory
            else:
                wi = W[indices[i],:]
                res[indices[i]] = self.process(wi)
        return (res)

class square_FT(polyFT):
    '''
    Derived class for a square mask.
    Initialization takes half size c as input
    '''

    def __init__(self,c):

        self.c = c
        Gamma = self.square_coordinates(self.c)
        super().__init__(Gamma)
        return

    def square_coordinates(self,c):

        arr = np.array(([c,c],[c,-c],[-c,-c],[-c,c]))
        return(arr)

    def square_transform(self,W):
        '''
        computes FT of square mask of half size c
        '''
        u = 2.*np.pi * W[:,0]
        v = 2.*np.pi * W[:,1]
        u0 = np.abs(u)<1e-10
        v0 = np.abs(v)<1e-10
        uv0 = u0*v0
        res = 4./(u*v)*np.sin(u*self.c)*np.sin(v*self.c)
        res[u0] = 4.*self.c/v[u0]*np.sin(v[u0]*self.c)
        res[v0] = 4.*self.c/u[v0]*np.sin(u[v0]*self.c)
        res[uv0] = 4.*self.c**2
        return(res)



class disk_FT (polyFT):

    '''
    Derived class for a circular mask.
    Initialization takes radius and number of points 
    for the polygonal approximation of the disk
    '''

    def __init__(self, n, R=1.0):

        self.R = R

        theta = np.arange(n)/n * 2.*np.pi
        Gamma = np.vstack((R*np.cos(theta),-R*np.sin(theta))).T
        super().__init__(Gamma)

        return

    def disk_transform(self,W):
        '''
        computes FT of disk of radius R
        '''
        rho = 2.*np.pi*np.linalg.norm(W,axis=1)
        res = 2.*np.pi*self.R**2 * sp.special.j1(rho*self.R) / (rho*self.R)
        res[rho<1e-10] = np.pi*self.R**2
        return(res)

    def pixelized_disk(self,n_pixels,n_pad):
        '''
        creates pixelized disk mask of radius R, with zero padding factor n_pad,
        and n_pixels on the side of the image.
        '''
        self.n_pixels = n_pixels
        self.n_pad = n_pad
        self.L = self.n_pad * self.R
        arr = np.fft.fftfreq(self.n_pixels,d=1./(2.*self.L))
        x, y = np.meshgrid(arr,arr)
        rxy = np.sqrt(x**2+y**2)
        mask = np.zeros((self.n_pixels,self.n_pixels))
        mask[rxy<self.R] = 1.0
        return (mask)

    def pixelized_bbox(self,upper=True):
        '''
        Computes the bounding box of the (centered) pixelized mask. 
        To be used in imshow routine with the "extent" keyword.
        upper=True gives the bounding box for origin='upper' in imshow
        '''
        if (self.n_pixels is None):
            print ('Call pixelized_mask method first.')
            return

        pixel_size = 2.*self.L / self.n_pixels
        if upper:
            extent = (-self.L-pixel_size/2., self.L-pixel_size/2.,self.L-pixel_size/2., -self.L-pixel_size/2. )
        else:
            extent = (-self.L-pixel_size/2., self.L-pixel_size/2.,-self.L-pixel_size/2., self.L-pixel_size/2. )
        return (extent)

    def pixelized_FT(self,n_pixels=2048, n_pad=2, return_W=True):

        mask = self.pixelized_disk(n_pixels,n_pad)
        fmask = np.fft.fftshift(np.fft.fft2(mask))
        # Normalize: divide by number of pixels, multiply by surface of image
        fmask /= self.n_pixels**2 / (2.*self.L)**2
        if (return_W):
            W = compute_W_array(n_pixels,step=2.*self.L/n_pixels)
            return (W, fmask)
        else:
            return (fmask)



class sampled_disk_FT:
    '''
    This class implements a discretized, sampled disk mask
    of a given radius and for a given number of pixels.
    Inputs are disk radius and linear pixel size of 2D array.
    Returns the 2D FFT of the array.
    '''

    def __init__(self, npixels, R=10.0):

        self.R = R
        self.npixels = npixels
        # Create mask array
        self.mask = np.zeros((self.npixels,self.npixels))
        # Compute cyclic coordinates
        x = np.outer(np.ones(self.npixels),np.fft.fftfreq(self.npixels)*self.npixels)
        y = x.T
        rad = np.sqrt(x**2+y**2)
        self.mask[rad < R] = 1.0
        return

    def __call__(self,return_W=True):
        
        '''
        Computes the 2D FFT of the pixelized mask
        '''
        if (return_W):
            W = compute_W_array(self.npixels)
        res = np.fft.fftshift(np.fft.fft2(self.mask))
        if (return_W):
            return(W,res)
        else:
            return(res)




class hexagon_FT(polyFT):

    '''
    Derived class for an hexagonal mask.
    Initialization takes outer radius as input
    '''

    def __init__ (self, R=1.0, **kwargs):

        self.R = R
        Gamma = self.hexagon_coordinates(self.R)
        super().__init__(Gamma,**kwargs)

        return

    def hexagon_coordinates(self,R):
        '''
        computes coordinates of hexagon vertices.
        R is outer circle radius
        '''
        r = np.sqrt(3.)/2. * R # inner radius
        Gamma = np.array([[R,0.],[R/2,-r],[-R/2,-r],[-R,0.],[-R/2,r],[R/2,r]])
        return (Gamma)

    def hexagon_transform(self,W):
        '''
        computes FT of hexagon mask of outer radius R.
        '''
        u = 2.*np.pi * W[:,0]
        v = 2.*np.pi * W[:,1]
        s3 = np.sqrt(3.)
        calc = -4*s3/(u+s3*v)/(u-s3*v)*np.cos(u*self.R)+ 2.*s3/u/(u+s3*v)*np.cos(u/2*self.R-s3/2*v*self.R) + 2.*s3/u/(u-s3*v)*np.cos(u/2*self.R+s3/2*v*self.R)
        return(calc)


class petal_FT(polyFT):

    '''
    Derived class for petal mask
    Initialization takes as inputs outer mask radius, outer radius, number of petals, 
    number of points per half petal border, and profile type
    '''

    def __init__(self, r_in = 1, r_out=2, n_petals=8, n_border = 100, profile_type='arch_cos', Gamma=None, **kwargs):

        '''
        Initializes petal_FT class, derived from poly_FT.
        Takes inner and outer radii (r_in, r_out) of extinction profile, number of petals, number
        of border sampling points per half petal, and type of profile as inputs.
        '''

        self.r_in = r_in
        self.r_out = r_out
        self.n_petals = n_petals
        self.n_border = n_border
        self.profile_type = profile_type
        self.n_pixels = None
        self.n_pad = None
        self.L = None

        if (self.profile_type=='sister'):
            if ('profile_path' not in kwargs.keys()):
                print('For a SISTER profile, needs MATLAB path to create the profile')
                return
            self.profile_path = kwargs['profile_path']
            if (not os.path.exists(self.profile_path)):
                print('SISTER profile path %s does not exist'%self.profile_path)
                return
            self.occ = loadmat(self.profile_path)
            # Squeeze occ['r'] and occ['Profile'] for further use
            self.occ['r'] = np.array(self.occ['r'].squeeze())
            self.occ['Profile'] = np.array(self.occ['Profile'].squeeze())
            #
            # Need to differentiate between r_last and r_out for SISTER profile...
            self.r_last = self.occ['r'][-1] # Last defined value of sampled SISTER profile
            self.r_out = float(self.occ['occulterDiameter']/2.)
            self.r_in  = self.r_out - float(self.occ['petalLength'])
            self.n_petals = int(self.occ['numPetals'])
        else:
            self.r_last = self.r_out # No SISTER nonsense for other profiles

        self.profile = self.create_profile()


        if (Gamma is None):
            Gamma = self.petal_coordinates()
                
        super().__init__(Gamma, **kwargs)

    def create_profile(self):
        if self.profile_type=='arch_cos':
            def arch_cos(r):
                '''
                Function that returns 1 till r_out/2, 0
                outside r_out, arch cosine betweeen r_out/2 and r_out
                '''
                r = np.atleast_1d(r)
                res = np.zeros_like(r)
                res [r<=self.r_in] = 1.0
                res [r>self.r_out] = 0.0
                ou = np.where((r>self.r_in)*(r<=self.r_out))
                res[ou] = np.cos((r[ou]-self.r_in)/(self.r_out-self.r_in) * np.pi)/2. + 0.5
                return(res)
            return (arch_cos)
        if self.profile_type=='sister':
            # Get infos and profile from Matlab file
            
            def sister(r):
                '''
                Function that does a linear interpolation of the sampled SISTER profile
                '''
                r = np.atleast_1d(r)
                if not np.all(r[:-1]<=r[1:]):
                    # Input must be sorted

                    iarg = np.argsort(r,axis=None) # Sort on flattened array, important if r is 2D
                    res = np.zeros_like(r)
                    if (r.ndim==2):
                        iarg = np.unravel_index(iarg,r.shape) # Get 2D index coordinates from flattened array indices
                    res[iarg] = np.interp(np.array(r[iarg]),np.array(self.occ['r']),np.array(self.occ['Profile']),right=0.0)
                else:
                    # Already sorted
                    res = np.interp(np.array(r),np.array(self.occ['r']),np.array(self.occ['Profile']),right=0.0)
                return(res)
            return (sister)


    def petal_coordinates(self, inverse_curvature=False, eps=1e-10):
        '''
        Computes coordinates of polygon summits on the petal borders.
        Makes sure singular points of the border are included
        eps is there to make sure last defined point is taken into account for SISTER profile
        '''
        
        # Here we use r_last for the radius of the outer singular points of the polygonal shape 
        r = np.linspace(self.r_last+eps,self.r_in,self.n_border)
        theta = self.profile(r) * np.pi / self.n_petals

        r = np.concatenate((np.flip(r)[1:-1],r))
        theta = np.concatenate((-np.flip(theta)[1:-1],theta))

        rr = r.copy()
        ttheta = theta.copy()

        for i in range(1,self.n_petals):
            rr = np.concatenate((rr,r))
            ttheta = np.concatenate((ttheta,theta + i*2.*np.pi/self.n_petals))

        # Put in clockwise order (counterclockwise as seen along +z)
        ttheta = np.flip(ttheta)
        rr = np.flip(rr)

        return (np.vstack((rr*np.cos(ttheta), rr*np.sin(ttheta))).T)

    def set_the_scene(self, embed_factor=4, margin=0.01):
        '''
        Compute L (Claude's notations)
        '''
        self.embed_factor = embed_factor
        self.margin = margin
        self.n_pad = self.embed_factor*(1.0 + self.margin)
        self.L = self.n_pad  * self.r_out # L for Claude

        return

    def pixelized_mask(self, n_pixels=2048, embed_factor=4, margin=0.01, inverted=True):
        '''
        Create digitized pixel mask or size n_pixels x n_pixels,
        of physical linear size r_out * n_pad
        '''
        self.n_pixels = n_pixels # N in Claude's notations
        self.set_the_scene(embed_factor=embed_factor,margin=margin)
        self.step = 2.*self.L / self.n_pixels
        arr = np.fft.fftfreq(self.n_pixels,d=1./(2.*self.L))
        x, y = np.meshgrid(arr,arr)
        rxy = np.sqrt(x**2+y**2)
        angxy = np.arctan2(y,x)
        Num = self.n_petals
        pxy = self.profile(rxy)
        neg=(Num*np.abs(np.mod(angxy+np.pi/Num,2*np.pi/Num)-np.pi/Num)/np.pi>pxy) + (rxy>=self.r_last) # r_last, not r_out
        if (inverted):
            return (1.0-neg)
        else:
            return(neg)

    def pixelized_bbox(self,upper=True):
        '''
        Computes the bounding box of the (centered) pixelized mask. 
        To be used in imshow routine with the "extent" keyword.
        upper=True gives the bounding box for origin='upper' in imshow
        '''
        if (self.L is None or self.n_pixels is None):
            print ('Call pixelized_mask method first.')
            return

        pixel_size = 2.*self.L / self.n_pixels
        if upper:
            extent = (-self.L-pixel_size/2., self.L-pixel_size/2.,self.L-pixel_size/2., -self.L-pixel_size/2. )
        else:
            extent = (-self.L-pixel_size/2., self.L-pixel_size/2.,-self.L-pixel_size/2., self.L-pixel_size/2. )
        return (extent)

    def pixelized_FT(self,n_pixels=2048, embed_factor=4, margin=0.01, inverted=True, return_W=True):
        '''
        Calls pixelized_mask and computes its 2D FFT.
        Optionally computes 2D array of frequencies
        ### BEWARE: mask needs to be centered on zero before FFT... TO BE DONE
        '''
        mask = self.pixelized_mask(n_pixels,embed_factor,margin,inverted)
        fmask = np.fft.fftshift(np.fft.fft2(mask))
        fmask /= self.n_pixels**2 / (2.*self.L)**2
        if (return_W):
            W = compute_W_array(n_pixels,step=2.*self.L/n_pixels)
            return (W, fmask)
        else:
            return (fmask)


def compute_W_array(n=1024,dims=2,step=1.0):
    '''
    computes 2D coordinates of spatial frequencies as a list of 2D vectors.
    For dims=1, computes a regular sampling of the v=0 line.
    For dims=2, computes a regular sampling of the uv plane.
    '''
    f = np.fft.fftshift(np.fft.fftfreq(n,d=step))
    if (dims==1):
        W = np.vstack((f,np.zeros_like(f))).T
        return(W)
    else:
        fxx,fyy = np.meshgrid(f,f)
        W = np.vstack((fxx.flatten(),fyy.flatten())).T
    return(W)




