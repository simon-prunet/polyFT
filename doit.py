from poly import *
from fresnel import diffraction
import numpy as np

# Create NW2 profile
n = 200
p = petal_FT(n_border=n,profile_type='sister',profile_path='Matlab_files/NW2.mat')

# Create diffraction instance
m = 2**13
diff = diffraction(p,m)
# Compute Fresnel filters
diff.compute_fresnel_filter()
# Compute polygonal mask transform
diff.compute_polygonal_fmask()
# Compute diffraction patterns
diff.compute_diffraction_patterns()
# Save polygonal mask and diffraction patterns
np.save('fmask_n%d_m%d'%(n,m),diff.polygonal_fmask)
# Save diffraction patterns
np.save('diffraction_n%d_m%d'%(n,m),diff.diffracted)

