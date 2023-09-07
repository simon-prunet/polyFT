###
### Diagonal of matrix product, used to compute element wise (2D) dot products on rows and columns, 
### is obtained in the following way: a_{ii} = \sum_j b_{ij} c^T_{ji} = \sum_j b_{ij} c_{ij}
### which in numpy language is written (B*C).sum(-1)


import numpy as np

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



def hexagon_coordinates(R=1.0):
	'''
	computes coordinates of hexagon vertices.
	R is outer circle radius
	'''
	r = np.sqrt(3.)/2. * R # inner radius
	Gamma = np.array([[R,0.],[R/2,r],[-R/2,r],[-R,0.],[-R/2,-r],[R/2,-r]])
	return (Gamma)

def hexagon_transform(W,R=1.0):
	'''
	computes FT of hexagon mask of outer radius R.
	'''
	u = 2.*np.pi * W[:,0]
	v = 2.*np.pi * W[:,1]
	s3 = np.sqrt(3.)
	calc = -4*s3/(u+s3*v)/(u-s3*v)*np.cos(u*R) + 2.*s3/u/(u+s3*v)*np.cos(0.5*u*R-s3/2.*v*R) + 2.*s3/u/(u-s3*v)*np.cos(u/2*R+s3/2*v*R)
	return(calc)
