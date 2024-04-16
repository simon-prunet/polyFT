from poly import *
from fresnel import diffraction
import numpy as np

# Create NW2 profile
n = 2000
print('Creating petal with %d points'%(n*2*24))
p = petal_FT(n_border=n,profile_type='sister',profile_path='Matlab_files/NW2.mat')

# Create diffraction instance
m = 2**13
diff = diffraction(p,m)

print('Computing polygonal mask transform')
diff.compute_polygonal_fmask()
