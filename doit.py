from poly import *
from fresnel import diffraction
import numpy as np

# Choose GPU
cp.cuda.Device(2).use()
# Create NW2 profile
n = 2000
print('Creating petal with %d points'%(n*2*24))
#p = petal_FT(n_border=n,profile_type='sister',profile_path='Matlab_files/NW2.mat')
p = petal_FT(n_border=n,profile_type='trapeze',profile_path='Matlab_files/Param_PourOc.mat')

# Create diffraction instance
m = 2**13
diff = diffraction(p,m)
# Compute Fresnel filters
print('Computing Fresnel filters')
diff.compute_fresnel_filter()
# Compute polygonal mask transform
print('Computing polygonal mask transform')
diff.compute_polygonal_fmask()
# Compute diffraction patterns
print('Computing diffraction patterns')
diff.compute_diffraction_patterns()
# Save polygonal mask and diffraction patterns
print('Saving mask transform')
#np.save('/data101/prunet/fmask_n%d_m%d'%(n,m),diff.polygonal_fmask)
np.save('/scratch/prunet/polyFT/trapeze/fmask_n%d_m%d'%(n,m),diff.polygonal_fmask)
# Save diffraction patterns
print('Saving diffraction patterns')
#np.save('/data101/prunet/diffraction_n%d_m%d'%(n,m),diff.diffracted)
np.save('/scratch/prunet/polyFT/trapeze/diffraction_n%d_m%d'%(n,m),diff.diffracted)

