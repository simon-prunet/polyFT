"""
polygon_fourier.py -- Continuous Fourier transform of polygonal masks (form factors).

Built on top of polygon.py: every class here inherits its geometry from a class
of polygon.py, and only adds the Fourier machinery (process, __call__, analytic
transforms, FFT-based pixelized transforms).

Design: the Fourier machinery lives in FourierMixin. A Fourier shape is just
    class SquareFT(FourierMixin, Square)
so the same recipe applies to any geometry class, and a Fresnel layer can do
    class SquareFresnel(FresnelMixin, Square)
without knowing anything about this module.

Note: diagonal of a matrix product, used to compute element wise (2D) dot
products on rows and columns, is obtained in the following way:
a_{ii} = \sum_j b_{ij} c^T_{ji} = \sum_j b_{ij} c_{ij}
which in numpy language is written (B*C).sum(-1)
"""

from scipy.special import j1

from polygon import (np, cp, cuda_on, get_array_module, rot,
                     Polygon, Square, Hexagon, Disk, Petal)


# ---------------------------------------------------------------------------
# Fourier machinery
# ---------------------------------------------------------------------------

class FourierMixin:
    '''
    Adds the Fourier transform of the indicatrix function of a polygonal surface,
    at an arbitrary set of positions in uv space.

    Requires the host class to provide the Polygon interface: Gamma, npoints,
    edges, midpoints, tangents, area().

    sinc_formula=True  : formula from J. Wuttke (arxiv:1703.00255, math-ph), using
                         half edge vectors and coordinates of edge middle.
    sinc_formula=False : formula from the 1983 original paper (does not handle the
                         zero frequency, nor frequencies orthogonal to an edge).
    '''

    def __init__(self, *args, sinc_formula=True, **kwargs):
        super().__init__(*args, **kwargs)   # geometry first
        self.sinc_formula = bool(sinc_formula)
        if self.sinc_formula:
            self.Ej = self.edges / 2.
            self.Rj = self.midpoints
        else:
            # Normalized polygone edges \alpha_n
            self.Alpha = self.tangents
            # Also keep shifted Alpha matrix handy
            self.Alpha_m1 = np.roll(self.Alpha,1,axis=0)
            self.num_weight = ( rot(self.Alpha)*self.Alpha_m1 ).sum(-1)

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

        if self.sinc_formula:
            Rj = xp.asarray(self.Rj)
            Ej = xp.asarray(self.Ej)
            wx = rot(w)

            num_weight = xp.exp(2j*xp.pi*xp.dot(w,Rj.T)) #phase term
            num_weight *= xp.sinc(2.*xp.dot(w,Ej.T)) # sinc term
            num_weight *= xp.dot(wx,Ej.T) # geometric term

            result = -num_weight.sum(-1) / xp.linalg.norm(w,axis=1)**2 / (1j*xp.pi) # 1/q^2 term
            # Take care of W=(0,0) null frequency case: result is polygone area
            result[xp.linalg.norm(w,axis=1)==0] = self.area()
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

    def __call__ (self,W,cpu_memory_limit=50,gpu_memory_limit=10,verbose=True):

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

        if verbose:
            print ('nslices = ',nslices)

        res = np.zeros(nw,dtype=np.complex128)
        indices = np.array_split(np.arange(nw),nslices)
        for i in range(nslices):
            if verbose:
                print ('Processing slice number %d out of %d'%(i,nslices))
            if (cuda_on):
                wi = cp.asarray(W[indices[i],:])
                if verbose:
                    print ('shape of wi is ',wi.shape)
                    print(cp._default_memory_pool.used_bytes())
                resi = self.process(wi)
                res[indices[i]] = cp.asnumpy(resi)
                del wi, resi # Clean GPU memory
            else:
                wi = W[indices[i],:]
                res[indices[i]] = self.process(wi)
        return (res)


# ---------------------------------------------------------------------------
# Helpers for FFT-based (pixelized) transforms
# ---------------------------------------------------------------------------

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


def mask_fft(mask, L, return_W=True):
    '''
    2D FFT of a pixelized mask of physical half size L, normalized to a
    continuous Fourier transform (divide by number of pixels, multiply by
    surface of image). Optionally returns the matching frequency list.
    '''
    n_pixels = mask.shape[0]
    fmask = np.fft.fftshift(np.fft.fft2(mask))
    fmask /= n_pixels**2 / (2.*L)**2
    if (return_W):
        W = compute_W_array(n_pixels,step=2.*L/n_pixels)
        return (W, fmask)
    else:
        return (fmask)


# ---------------------------------------------------------------------------
# Generic polygon and specific shapes
# ---------------------------------------------------------------------------

class PolygonFT(FourierMixin, Polygon):
    '''
    Fourier transform of an arbitrary polygonal mask.
    Takes as input the 2D coordinates of the polygone summits.
    '''

    def __init__(self, Gamma, sinc_formula=True):
        super().__init__(Gamma, sinc_formula=sinc_formula)


class SquareFT(FourierMixin, Square):
    '''
    Square mask of half size c, with its analytic transform for comparison.
    '''

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


class HexagonFT(FourierMixin, Hexagon):
    '''
    Hexagonal mask of outer radius R, with its analytic transform for comparison.
    '''

    def hexagon_transform(self,W):
        '''
        computes FT of hexagon mask of outer radius R.
        '''
        u = 2.*np.pi * W[:,0]
        v = 2.*np.pi * W[:,1]
        s3 = np.sqrt(3.)
        calc = -4*s3/(u+s3*v)/(u-s3*v)*np.cos(u*self.R)+ 2.*s3/u/(u+s3*v)*np.cos(u/2*self.R-s3/2*v*self.R) + 2.*s3/u/(u-s3*v)*np.cos(u/2*self.R+s3/2*v*self.R)
        return(calc)


class DiskFT(FourierMixin, Disk):
    '''
    Circular mask (polygonal approximation), with its analytic (Airy)
    transform and FFT-based transform of the pixelized disk.
    '''

    def disk_transform(self,W):
        '''
        computes FT of disk of radius R
        '''
        rho = 2.*np.pi*np.linalg.norm(W,axis=1)
        res = 2.*np.pi*self.R**2 * j1(rho*self.R) / (rho*self.R)
        res[rho<1e-10] = np.pi*self.R**2
        return(res)

    def pixelized_FT(self,n_pixels=2048, n_pad=2, return_W=True):
        mask = self.pixelized_disk(n_pixels,n_pad)
        return mask_fft(mask, self.L, return_W=return_W)


class PetalFT(FourierMixin, Petal):
    '''
    Petal mask: polygonal transform plus FFT-based transform of the pixelized mask.
    Constructor arguments are those of Petal, plus sinc_formula.
    '''

    def pixelized_FT(self,n_pixels=2048, embed_factor=4, margin=0.01, inverted=True, return_W=True):
        '''
        Calls pixelized_mask and computes its 2D FFT.
        Optionally computes 2D array of frequencies
        ### BEWARE: mask needs to be centered on zero before FFT... TO BE DONE
        '''
        mask = self.pixelized_mask(n_pixels,embed_factor,margin,inverted)
        return mask_fft(mask, self.L, return_W=return_W)


class SampledDiskFT:
    '''
    This class implements a discretized, sampled disk mask
    of a given radius and for a given number of pixels.
    Inputs are disk radius and linear pixel size of 2D array.
    Returns the 2D FFT of the array.
    (Pure FFT object: no polygon involved, hence no geometry base class.)
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


# ---------------------------------------------------------------------------
# Backward-compatible names (old poly.py)
# ---------------------------------------------------------------------------

polyFT = PolygonFT
square_FT = SquareFT
hexagon_FT = HexagonFT
disk_FT = DiskFT
petal_FT = PetalFT
sampled_disk_FT = SampledDiskFT
