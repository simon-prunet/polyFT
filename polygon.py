"""
polygon.py -- Polygon geometry (no Fourier, no Fresnel).

This module is the stable base layer. It knows about vertices, edges,
orientation, area, normals, pixelized masks... and nothing about what is
computed on top of them. Both the Fourier layer (polygon_fourier.py) and the
Fresnel layer inherit from the classes defined here.

Conventions
-----------
* Gamma is an (n, 2) array of vertex coordinates, in the order in which the
  polygon boundary is traversed. The polygon is closed implicitly: edge j goes
  from Gamma[j] to Gamma[j+1] (cyclically).
* The shapes below all use CLOCKWISE ordering (in standard x-right / y-up axes),
  for which area() is positive. This is the orientation the form-factor
  formulas of polygon_fourier.py are written for.
* Outward normals are computed for either orientation.
* Geometric quantities (edges, lengths, tangents, normals...) are lazily
  computed and cached. Treat Gamma as immutable once the object is built.

BEWARE (petal masks)
All things related to the radial profile use r_last (last defined point in
SISTER profile), including contour samples for the polygonal transform.
BUT all pixel/frequency axes use r_out = occulterDiameter/2
"""

import os
from functools import cached_property

try:
    import cupy as cp
    import numpy as np
    cuda_on = True
except Exception:
    import numpy as np
    cp = None
    cuda_on = False

from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Small array helpers (numpy / cupy agnostic)
# ---------------------------------------------------------------------------

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
    '''
    Rotates each 2D row vector by +90 degrees: (x, y) -> (-y, x)
    '''
    xp = get_array_module(arr)
    return xp.vstack((-arr[:,1],arr[:,0])).T


# ---------------------------------------------------------------------------
# Generic polygon
# ---------------------------------------------------------------------------

class Polygon:
    '''
    Arbitrary closed polygon, defined by the 2D coordinates of its summits.
    Pure geometry: everything a Fourier or Fresnel layer needs to know about
    the boundary is exposed here.
    '''

    def __init__(self, Gamma):
        Gamma = np.array(Gamma, dtype=float)
        if Gamma.ndim != 2 or Gamma.shape[1] != 2 or Gamma.shape[0] < 3:
            raise ValueError('Gamma must be an (n, 2) array with n >= 3, got shape %s' % (Gamma.shape,))
        self.Gamma = Gamma
        self.npoints = Gamma.shape[0]   # number of vertices == number of edges

    # -- edges ------------------------------------------------------------

    @cached_property
    def next_vertices(self):
        '''Gamma_{j+1}, cyclically'''
        return np.roll(self.Gamma,-1,axis=0)

    @cached_property
    def edges(self):
        '''Edge vectors E_j = Gamma_{j+1} - Gamma_j, shape (n, 2)'''
        return self.next_vertices - self.Gamma

    @cached_property
    def midpoints(self):
        '''Edge middle points (Gamma_j + Gamma_{j+1})/2, shape (n, 2)'''
        return (self.next_vertices + self.Gamma) / 2.

    @cached_property
    def lengths(self):
        '''Edge lengths, shape (n,)'''
        return np.linalg.norm(self.edges,axis=1)

    @property
    def perimeter(self):
        return self.lengths.sum()

    @cached_property
    def tangents(self):
        '''Unit tangent vectors along the traversal direction, shape (n, 2)'''
        return self.edges / self.lengths[:,None]

    @cached_property
    def normals(self):
        '''
        Unit OUTWARD normals of each edge, shape (n, 2).
        For a clockwise polygon, the interior is on the right of the direction
        of travel, so the outward normal is the tangent rotated by +90 degrees.
        '''
        sign = 1.0 if self.is_clockwise else -1.0
        return sign * rot(self.tangents)

    def iter_edges(self):
        '''Yields (start_vertex, end_vertex) for each edge'''
        for a, b in zip(self.Gamma, self.next_vertices):
            yield a, b

    def edge_points(self, t):
        '''
        Points at parameter t in [0, 1] along every edge:
        Gamma_j + t * E_j. Returns an array of shape (n_edges, len(t), 2).
        '''
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.Gamma[:,None,:] + t[None,:,None] * self.edges[:,None,:]

    # -- area, orientation, centroid -----------------------------------------

    def area (self):
        '''
        Polygonal area (zero-frequency term of the Fourier transform).
        SIGNED: positive for the clockwise ordering used throughout this code,
        negative for counterclockwise.
        '''
        Gamma = self.Gamma
        res = 0.5 * np.sum(-np.roll(rot(Gamma),1,axis=0)*Gamma) # \sum [\hat{n},V_{j-1},V_{j}]
        return (res)

    @property
    def is_clockwise(self):
        return bool(self.area() > 0)

    @property
    def centroid(self):
        '''Center of mass of the polygonal surface (independent of orientation)'''
        G, Gn = self.Gamma, self.next_vertices
        cross = G[:,0]*Gn[:,1] - Gn[:,0]*G[:,1]
        A = 0.5 * cross.sum()
        return ((G + Gn) * cross[:,None]).sum(0) / (6.*A)


# ---------------------------------------------------------------------------
# Pixelization helper shared by disk and petal masks
# ---------------------------------------------------------------------------

class PixelizedMaskMixin:
    '''
    Bounding box of a centered pixelized mask. Classes using it must set
    n_pixels and L (half size of the pixel array, physical units) when they
    build their pixelized mask.
    '''
    n_pixels = None
    n_pad = None
    L = None

    def pixelized_bbox(self,upper=True):
        '''
        Computes the bounding box of the (centered) pixelized mask.
        To be used in imshow routine with the "extent" keyword.
        upper=True gives the bounding box for origin='upper' in imshow
        '''
        if (self.L is None or self.n_pixels is None):
            raise RuntimeError('Build the pixelized mask first.')

        pixel_size = 2.*self.L / self.n_pixels
        if upper:
            extent = (-self.L-pixel_size/2., self.L-pixel_size/2.,self.L-pixel_size/2., -self.L-pixel_size/2. )
        else:
            extent = (-self.L-pixel_size/2., self.L-pixel_size/2.,-self.L-pixel_size/2., self.L-pixel_size/2. )
        return (extent)


# ---------------------------------------------------------------------------
# Specific shapes
# ---------------------------------------------------------------------------

class Square(Polygon):
    '''
    Square mask. Initialization takes half size c as input
    '''

    def __init__(self,c):
        self.c = c
        super().__init__(self.square_coordinates(self.c))

    def square_coordinates(self,c):
        arr = np.array(([c,c],[c,-c],[-c,-c],[-c,c]))
        return(arr)


class Hexagon(Polygon):
    '''
    Hexagonal mask. Initialization takes outer radius as input
    '''

    def __init__ (self, R=1.0):
        self.R = R
        super().__init__(self.hexagon_coordinates(self.R))

    def hexagon_coordinates(self,R):
        '''
        computes coordinates of hexagon vertices.
        R is outer circle radius
        '''
        r = np.sqrt(3.)/2. * R # inner radius
        Gamma = np.array([[R,0.],[R/2,-r],[-R/2,-r],[-R,0.],[-R/2,r],[R/2,r]])
        return (Gamma)


class Disk(PixelizedMaskMixin, Polygon):
    '''
    Circular mask, approximated by a regular polygon.
    Initialization takes the number of points for the polygonal
    approximation of the disk, and its radius.
    '''

    def __init__(self, n, R=1.0):
        self.R = R
        theta = np.arange(n)/n * 2.*np.pi
        Gamma = np.vstack((R*np.cos(theta),-R*np.sin(theta))).T
        super().__init__(Gamma)

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


class Petal(PixelizedMaskMixin, Polygon):
    '''
    Petal mask.
    Initialization takes as inputs inner and outer radii of the extinction
    profile, number of petals, number of points per half petal border, and
    profile type ('arch_cos', 'sister', 'trapeze', 'serrated').
    'sister' and 'trapeze' also need profile_path=<MATLAB file>.
    '''

    def __init__(self, r_in = 1, r_out=2, n_petals=8, n_border = 100, profile_type='arch_cos', Gamma=None, **kwargs):

        self.r_in = r_in
        self.r_out = r_out
        self.n_petals = n_petals
        self.n_border = n_border
        self.profile_type = profile_type

        if (self.profile_type=='sister'):
            self.profile_path = self._get_profile_path(kwargs, 'SISTER')
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

        elif (self.profile_type=='trapeze'):
            self.profile_path = self._get_profile_path(kwargs, 'TRAPEZE')
            self.occ = loadmat(self.profile_path)
            # Added by hand for now. Will need to be included later in .mat file
            self.occ['Z'] = np.array([[80000000]]) #80000 km
            self.occ['lambdaRange'] = np.array([[0.65e-6]])

            self.r_in = float(self.occ['RayMinOc'])
            self.r_out = float(self.occ['RayMaxOc'])
            self.r_last = self.r_out # only different for SISTER profile
            self.alphas = self.occ['alpha'].squeeze()
            self.n_trapeze = int(self.occ['Nbre_Trapez'])
            self.n_pupil = int(self.occ['NptsPup'])
            self.positions = np.linspace(self.r_in,self.r_out,self.n_trapeze+1)

            # Precompute profile at self.positions
            self.sampled_profile = np.zeros(self.n_trapeze+1)
            self.sampled_profile[:-1] = np.dot(np.triu(np.ones((self.n_trapeze,self.n_trapeze))), self.alphas)
            # Make sure that first value is exactly 1 (at r_min)
            self.sampled_profile[0] = 1.0

        elif (self.profile_type=='linearly_interpolated'):
            self.profile_path = self._get_profile_path(kwargs, 'LINEARLY_INTERPOLATED')
            self.occ = loadmat(self.profile_path)
            # Add these parameters by hand for now... Should be included in the .mat file
            self.occ['Z'] = np.array([[80000000]]) # 80000 km
            self.occ['lambdaRange'] = np.array([[0.65e-6]])
        
            self.r_in = 10.0
            self.r_out = 25.0
            self.r_last = self.r_out # only different for SISTER profile
            self.sampled_profile = self.occ['profil'].squeeze()
            self.n_points = self.sampled_profile.size
            # Beware ! Profile starts at r=0...
            # self.positions = np.linspace(0.0,self.r_out,self.n_points)
            self.positions = self.occ['axeR'].squeeze()
            
        elif (self.profile_type=='serrated'):
            self.n_border = 2*self.n_petals
            self.r_last = self.r_out

        elif (self.profile_type=='arch_cos'):
            self.r_last = self.r_out # Only different for SISTER profile

        else:
            raise ValueError('Unknown profile_type %r' % self.profile_type)

        # Profile function not needed for serrated mask, but needed for other profiles
        if (self.profile_type!='serrated'):
            self.profile = self.create_profile()
            if (Gamma is None):
                Gamma = self.petal_coordinates_from_profile()
        else:
            if (Gamma is None):
                Gamma = self.petal_coordinates_serrated()

        super().__init__(Gamma)

    @staticmethod
    def _get_profile_path(kwargs, label):
        if ('profile_path' not in kwargs):
            raise ValueError('For a %s profile, needs MATLAB path to create the profile' % label)
        path = kwargs['profile_path']
        if (not os.path.exists(path)):
            raise FileNotFoundError('%s profile path %s does not exist' % (label, path))
        return path

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
        if self.profile_type=='trapeze':

            def trapeze_profile(r):
                '''
                Function that computes the weighted sum of trapeze
                '''
                # Now compute profile values at self.positions.
                # Note that at self.r_min, profile values is \sum\alpha=1, at self.positions[-2]: \alpha[-1], at self.r_max: 0
                #
                res = np.interp(np.array(r),self.positions,self.sampled_profile, left=1.0, right=0.0)
                return(res)
            return (trapeze_profile)

        if self.profile_type=='linearly_interpolated':

            def piecewise_linear_profile(r):
                '''
                Function that linearly interpolates sampled profile
                '''
                res = np.interp(np.array(r),self.positions,self.sampled_profile, right=0.0)
                return(res)
            return (piecewise_linear_profile)

        raise ValueError('No profile function for profile_type %r' % self.profile_type)

    def petal_coordinates_from_profile(self, inverse_curvature=False, eps=1e-10):
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

    def petal_coordinates_serrated(self):
        '''
        Computes coordinates of polygon summits on the serrated petal borders.
        '''
        r = np.array([self.r_out, self.r_in])
        theta = np.array([0., np.pi/self.n_petals])

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
