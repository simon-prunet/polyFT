from polygon import (np, cp, cuda_on, get_array_module, rot,
                     Polygon, Square, Hexagon, Disk, Petal)

class FresnelMixin:

    '''
    Mixin class for computation of Fresnel integrals on polygonal apertures. 
    Computations are done on the boundary, as in Maggi-Rubinowicz theory.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_filter = phasefilter(self,m=self.m)
        self.W = compute_W_array(self.m, step=2.*self.L/self.m)
        return