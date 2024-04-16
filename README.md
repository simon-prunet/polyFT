# polyFT
2D Fourier transform of indicator function of polygonal mask. 

This code is based on formulas derived by Lee, S.-W. & Mittra, R. (1983), and further by Wuttke, J. (2021).
The purpose of this code is to compute so-called polygonal shape factors, which correspond to 2D, continuous Fourier transforms
of indicator functions of polygonal shapes, as discrete sums over the polygone vertices.

First derived in the context of radio receivers (Lee, S.-W. & Mittra, R., 1983), and further developped in the context of
X-ray and Gamma diffraction (Wuttke, J. 2021), the formulas that follow are here implemented in a simple, stand-alone python code
that can make use of gpu acceleration via cupy if available, or otherwise defaults to numpy operations.
Care is taken to control the memory footprint at a basic level.

These transforms have been used in the context of Fresnel diffraction by binary masks, with polygones of up to ~1M vertices.

