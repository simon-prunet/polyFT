
try:
	import cupy as np
except:
	import numpy as np

from scipy.io import loadmat
import scipy as sp

import poly

class occulter:

	''' 
	This class creates an occulter from a SISTER transmission profile,
	number of petals, physical size, padding region, etc.
	'''
	def __init__(self,path='Matlab_files/NW2.mat', **kwargs):

		''' 
		At initialization, we will load the content of the SISTER Matlab file
		'''
		dic = loadmat(path)
		self.dic = dic

	def profile(self,r):
		'''
		Interpolate SISTER profile on array of radii r
		'''
		if not np.all(r[:-1] <= r[1:]):
			print ('Array of coordinates needs to be sorted in interpolation')
			return None
		res = np.interp(r,self.dic['r'].squeeze(),self.dic['Profile'].squeeze(),right=0.0)
		return(res)




