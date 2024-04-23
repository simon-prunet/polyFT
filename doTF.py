from poly import *
from fresnel import diffraction
import numpy as np
from timer import Timer

# Create NW2 profile
n = 8000
print('Creating petal with %d points'%(n*2*24))
p = petal_FT(n_border=n,profile_type='sister',profile_path='Matlab_files/NW2.mat')

# Create diffraction instance
m = 2**10
diff = diffraction(p,m)

t = Timer()
print('Computing polygonal mask transform')
t.start()
diff.compute_polygonal_fmask(gpu_memory_limit=30)
t.stop()
