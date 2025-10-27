import numpy as np
from ..util.backend_functions import backend as bd
import numpy as np
from .diffractive_element import DOE

class RingSlit(DOE):
    def __init__(self, outer_radius, thickness, x0=0, y0=0):
        """
        Creates a ring slit centered at (x0, y0) with given outer radius and thickness.
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.x0 = x0
        self.y0 = y0
        self.outer_radius = outer_radius
        self.thickness = thickness
        self.inner_radius = max(0, outer_radius - thickness)

    def get_transmittance(self, xx, yy, λ):
        rr2 = (xx - self.x0) ** 2 + (yy - self.y0) ** 2
        t = bd.select(
            [
                (rr2 < self.outer_radius ** 2) & (rr2 >= self.inner_radius ** 2),
                bd.ones_like(xx, dtype=bool)
            ],
            [
                bd.ones_like(xx),
                bd.zeros_like(xx)
            ]
        )
        return t

    def get_coherent_PSF(self, xx, yy, z, λ):
        """
        Get the coherent point spread function (PSF) of the DOE when it acts as the pupil of an imaging system.
        For a ring slit, the PSF is the difference between two circular apertures.
        """
        if bd == np:
            from scipy import special
        else:
            from cupyx.scipy import special

        rr = bd.sqrt(xx**2 + yy**2)
        tmp_outer = 2 * bd.pi * self.outer_radius * rr / (λ * z)
        tmp_inner = 2 * bd.pi * self.inner_radius * rr / (λ * z)
        tmp_outer = bd.where(tmp_outer < 1e-9, 1e-9, tmp_outer)
        tmp_inner = bd.where(tmp_inner < 1e-9, 1e-9, tmp_inner)

        PSF_outer = 2 * bd.pi * self.outer_radius**2 * (special.j1(tmp_outer)) / tmp_outer
        PSF_inner = 2 * bd.pi * self.inner_radius**2 * (special.j1(tmp_inner)) / tmp_inner
        PSF = 1 / (z * λ) ** 2 * (PSF_outer - PSF_inner)

        return PSF