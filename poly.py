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
		self.Alpha = Gamma - np.roll(Gamma,1,axis=0)
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

		den_weight = np.dot(W,self.Alpha.T) * np.dot(W,self.Alpha_m1.T)
		phase = np.exp(2j*np.pi*np.dot(W,self.Gamma.T))
		return (phase * self.num_weight[None,:]/den_weight).sum(-1)



