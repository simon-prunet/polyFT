# Similar to the notebook doFr.ipynb
# To be run on the server with GPU

import numpy as np
from polygon_fresnel import *

# Parameters ####
dims = 1
m = 2**19
r_in = 0.71 # metres
r_out = 0.76 #metres
Z = 144 # metres
ll = np.array([550e-9]) # metres
n_petals = 4

# Create serrated Fresnel structure ####
s = SerratedFresnel(Z=Z, lambdas=ll, n_petals=n_petals,r_in=r_in,r_out=r_out); 
s.set_the_scene(embed_factor=1.2)

# Create array of target points in pupil plane, compute diffracted amplitude ####
P = compute_P_array(m,step=1./(2*s.L), dims=dims)
res = s(P , verbose=True)

# Compute geometric optics contribution ####
out = s.is_outside_of_mask(P)
res_tot = res.copy()
res_tot[out] += 1. # Add geometric optics part

np.savez('Serrated_radius_npetals_%d.npz'%n_petals, P, res_tot)

