###
### Diagonal of matrix product, used to compute element wise (2D) dot products on rows and columns, 
### is obtained in the following way: a_{ii} = \sum_j b_{ij} c^T_{ji} = \sum_j b_{ij} c_{ij}
### which in numpy language is written (B*C).sum(-1)


import numpy as np
import scipy as sp

def rot(arr):
    return np.vstack((-arr[:,1],arr[:,0])).T

class polyFT:
	
	def __init__(self, Gamma):

		'''
		Initializes the polygonal Fourier Transform class.
		This will allow to compute the Fourier Transform of the indicatrix 
		function of an arbitrary polygonal surface, at an arbitrary set of 
		positions in uv space.

		Takes as input the 2D coordinates of the polygone summits.
		'''

		self.Gamma = Gamma
		self.npoints = Gamma.shape[0]
		# Compute normalized polygone edges \alpha_n
		#self.Alpha = Gamma - np.roll(Gamma,1,axis=0)
		self.Alpha = np.roll(Gamma,-1,axis=0) - Gamma
		self.Alpha /= np.linalg.norm(self.Alpha,axis=1)[:,None]
		# Also keep shifted Alpha matrix handy
		self.Alpha_m1 = np.roll(self.Alpha,1,axis=0)
		self.num_weight = ( rot(self.Alpha)*self.Alpha_m1 ).sum(-1)

		return

	def __call__(self,W):
		
		'''
		Computes Fourier transform of indicatrix function of polygonal shape,
		at 2D positions in Fourier space specified by the W matrix
		'''

		# Beware that W are wave vectors in the paper, but are assumed to be spatial frequencies here,
		# to allow direct comparisons to FFT computations, hence the 2pi factors in denominator and phase.
		# Note that this is purely conventional.
		den_weight = np.dot(W,self.Alpha.T) * np.dot(W,self.Alpha_m1.T) * (2.*np.pi)**2
		phase = np.exp(2j*np.pi*np.dot(W,self.Gamma.T))
		return (phase * self.num_weight[None,:]/den_weight).sum(-1)


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

	def __init__ (self, R=1.0):

		self.R = R
		Gamma = self.hexagon_coordinates(self.R)
		super().__init__(Gamma)

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



def compute_W_array(n=1024,dims=2):
	'''
	computes 2D coordinates of spatial frequencies as a list of 2D vectors.
	For dims=1, computes a regular sampling of the v=0 line.
	For dims=2, computes a regular sampling of the uv plane.
	'''
	f = np.fft.fftshift(np.fft.fftfreq(n))
	if (dims==1):
		W = np.vstack((f,np.zeros_like(f))).T
		return(W)
	else:
		fxx,fyy = np.meshgrid(f,f)
		W = np.vstack((fxx.flatten(),fyy.flatten())).T
	return(W)




