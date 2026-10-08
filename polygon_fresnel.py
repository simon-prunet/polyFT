from polygon import (np, cp, cuda_on, get_array_module, rot,
                     Polygon, Square, Hexagon, Disk, Petal)

#from psi_gpu import (occulter_edge_integral_batch)
from occulter_gpu import (occulter_edge_integral)

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
        self.qj = xp.asarray(self.Gamma)  # Vertices of the polygon
        self.nj = xp.asarray(self.normals)  # Normals to the edges of the polygon
        self.tj = xp.asarray(self.tangents)  # Tangents to the edges of the polygon
        self.lj = xp.asarray(self.lengths)  # Lengths of the edges of the polygon

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

    def process(self, p, j, i_lambda, order=48):
        '''
            This method computes the Fresnel diffraction pattern for the polygonal aperture defined by the vertices qj, normals nj, and tangents tj. 
            The computation is performed using the Fresnel integrals along the edges of the polygon.
            p : array of points in the observation plane where the diffraction pattern is computed.
            j : index of the edge for which the computation is performed.
            i_lambda : index of the wavelength in self.lambdaRange for which the computation is performed.
        '''

        ## This does not have the right dimensions !!! Needs to be fixed
        qj = self.qj[j]  # Vertex of the j-th edge
        nj = self.nj[j]  # Normal to the j-th edge
        tj = self.tj[j]  # Tangent to the j-th edge
        lj = self.lj[j]  # Length of the j-th edge
        print('qj, nj,tj,lj',qj, nj,tj,lj)

        lammda = self.lambdaRange[i_lambda]  # Wavelength for the computation
        z = self.Z  # Propagation distance
    
        N = xp.dot(qj[None,:] - p[:,:], nj) * xp.sqrt(xp.pi / lammda / z) 
        T = xp.dot(qj[None,:] - p[:,:], tj) * np.sqrt(np.pi / lammda / z)
        Tp = T + xp.sqrt(xp.pi / lammda / z) * lj  # T plus edge lengths
        print('N,T,Tp',N.shape,T.shape,Tp.shape)


        res = N * np.exp(1j * N**2) * occulter_edge_integral(T, Tp, xp.abs(N), order=order) / (2.*np.pi) # Compute the Fresnel integral along the edges
        return(res)

    def __call__(self, P, verbose=True, order=48):
        '''
            This method computes the Fresnel diffraction pattern for the polygonal aperture at the points P in the observation plane.
            P : array of points in the observation plane where the diffraction pattern is computed.
        '''
        
        npupil = P.shape[0]
        res = np.zeros((npupil, self.lambdaRange.size), dtype='complex128')
        p = xp.asarray(P)

        for i_lambda in range(self.lambdaRange.size):
            for j in range(self.npoints):
                if verbose:
                    print ('Processing edge number %d out of %d'%(j+1,self.npoints))
                if (cuda_on):
                    if verbose:
                        print ('shape of p is ',p.shape)
                        print(xp._default_memory_pool.used_bytes())
                    resi = self.process(p, j, i_lambda, order=order)
                    res[:, i_lambda] += xp.asnumpy(resi) # Add contributions from all edges for the current wavelength
                    del resi # Clean GPU memory
                else:
                    res[:, i_lambda] += self.process(p, j, i_lambda)

        return res

############################################################################
# Helper functions for Fresnel integrals
############################################################################

def compute_P_array(m=2**12,dims=2,step=1.0):
    '''
    computes 2D coordinates of pupil samples as a list of 2D vectors.
    For dims=1, computes a regular sampling of the y=0 line.
    For dims=2, computes a regular sampling of the pupil plane.
    '''
    f = np.fft.fftshift(np.fft.fftfreq(m,d=step))
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