from polygon import (np, cp, cuda_on, get_array_module, rot,
                     Polygon, Square, Hexagon, Disk, Petal)

from psi_gpu import (occulter_edge_integral_batch)

try:
    import cupy as xp
    cuda_on = True
except ImportError:
    import numpy as xp
    cuda_on = False


class FresnelMixin:

    '''
    Mixin class for computation of Fresnel integrals on polygonal apertures. 
    Computations are done on the boundary, as in Maggi-Rubinowicz theory.
    '''

    def __init__(self, *args, lambdas = np.array([650e-9]), Z = 1., **kwargs):
        super().__init__(*args, **kwargs)
        self.qj = self.Gamma  # Vertices of the polygon
        self.nj = self.normals  # Normals to the edges of the polygon
        self.tj = self.tangents  # Tangents to the edges of the polygon
        if hasattr(self, 'occ'):
            if hasattr(self.occ, 'lambdaRange'):
                self.lambdaRange = self.occ.lambdaRange
        else:
            self.lambdaRange = lambdas
        if hasattr(self, 'occ'):
            if hasattr(self.occ, 'Z'):
                self.Z = self.occ.Z
        else:
            self.Z = Z  # Default propagation distance

    def process(self, p, i_lambda):
        '''
            This method computes the Fresnel diffraction pattern for the polygonal aperture defined by the vertices qj, normals nj, and tangents tj. 
            The computation is performed using the Fresnel integrals along the edges of the polygon.
            p : array of points in the observation plane where the diffraction pattern is computed.
            i_lambda : index of the wavelength in self.lambdaRange for which the computation is performed.
        '''
        # Create arrays (qj-p).tj, (qj-p).nj

        qj = xp.asarray(self.qj)  # Vertices of the polygon
        tj = xp.asarray(self.tj)  # Tangents to the edges of the polygon
        pp = xp.asarray(p)  # Points in the observation plane
        lengths = xp.asarray(self.lengths)  # Lengths of the edges of the polygon

        ## This does not have the right dimensions !!! Needs to be fixed
        N = xp.dot(self.qj[None,:,:] - pp[:,None,:], self.nj.T) * np.sqrt(np.pi / self.lambdaRange[i_lambda] / self.Z)
        T = xp.dot(self.qj[None,:,:] - pp[:,None,:], self.tj.T) * np.sqrt(np.pi / self.lambdaRange[i_lambda] / self.Z)
        Tp = T + np.sqrt(np.pi / self.lambdaRange[i_lambda] / self.Z) * lengths[None,:]  # T plus edge lengths

        res = N * np.exp(1j * N**2) * occulter_edge_integral_batch(T, Tp, xp.abs(N))  # Compute the Fresnel integral along the edges
        return(res.sum(axis=1))  # Sum over edges to get the total field at each point p

    def __call__(self, P, cpu_memory_limit=50,gpu_memory_limit=10,verbose=True):
        '''
            This method computes the Fresnel diffraction pattern for the polygonal aperture at the points P in the observation plane.
            P : array of points in the observation plane where the diffraction pattern is computed.
        '''
        
        npupil = P.shape[0]
        res = xp.zeros((npupil, self.lambdaRange.size), dtype='complex128')
        cpu_limit = cpu_memory_limit * 1024**3
        gpu_limit = gpu_memory_limit * 1024**3
        if  (cuda_on):
            nslices = npupils * self.npoints * self.order * 10 // gpu_limit + 1
        else:
            nslices = npupils * self.npoints * self.order * 10 // cpu_limit + 1

        if verbose:
            print ('nslices = ',nslices)

        res = np.zeros((npupil, self.lambdaRange.size), dtype=np.complex128)
        indices = np.array_split(np.arange(npupil),nslices)

        for i_lambda in range(self.lambdaRange.size):
            for i in range(nslices):
                if verbose:
                    print ('Processing slice number %d out of %d'%(i,nslices))
                if (cuda_on):
                    p = xp.asarray(P[indices[i],:])
                    if verbose:
                        print ('shape of p is ',p.shape)
                        print(xp._default_memory_pool.used_bytes())
                    resi = self.process(p, i_lambda)
                    res[indices[i], i_lambda] = xp.asnumpy(resi)
                    del p, resi # Clean GPU memory
                else:
                    p = P[indices[i],:]
                    res[indices[i], i_lambda] = self.process(p, i_lambda)
            
        return res

############################################################################
# Helper functions for Fresnel integrals
############################################################################

def compute_P_array(n=1024,dims=2,step=1.0):
    '''
    computes 2D coordinates of pupil samples as a list of 2D vectors.
    For dims=1, computes a regular sampling of the y=0 line.
    For dims=2, computes a regular sampling of the pupil plane.
    '''
    f = np.fft.fftshift(np.fft.fftfreq(n,d=step))
    if (dims==1):
        P = np.vstack((f,np.zeros_like(f))).T
        return(P)
    else:
        fxx,fyy = np.meshgrid(f,f)
        P = np.vstack((fxx.flatten(),fyy.flatten())).T
    return(P)

############################################################################
# Mixin class for Fresnel diffraction of polygonal apertures
############################################################################
class PolygonFresnel(FresnelMixin, Polygon):
    '''
    Fresnel diffraction of an arbitrary polygonal mask.
    Takes as input the 2D coordinates of the polygone summits.
    '''

    def __init__(self, Gamma, sinc_formula=True):
        super().__init__(Gamma, sinc_formula=sinc_formula)

class SquareFresnel(FresnelMixin, Square):
    '''
    Fresnel diffraction of a square mask.
    Takes as input the size of the square.
    '''

    def square_transform(self, P):
        '''
        Computes the Fresnel diffraction pattern of a square mask at the points P in the observation plane.
        P : array of points in the observation plane where the diffraction pattern is computed.
        '''
        return self.process(P, 0)  # Use the first wavelength in lambdaRange