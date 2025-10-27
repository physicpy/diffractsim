import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class DammannGratingOdd(DOE):
    def __init__(self, tx, ty, px, py,
                 width, height,
                 x0=0, y0=0,
                 absorbance=0):
        """
        Creates a 2D Dammann grating (amplitude + phase) at (x0,y0),
        with transition arrays tx, ty (normalized 0~1), periods px, py,
        and an absorbance (0–1) for the amplitude modulation.
        The grating is cropped to a rectangle of size (width×height).
        """
        global bd
        # import inside to ensure backend override works
        from ..util.backend_functions import backend as bd

        self.tx = np.sort(np.array(tx))  # sorted transition fractions
        self.ty = np.sort(np.array(ty))
        self.px = px
        self.py = py

        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0

        self.absorbance = absorbance

    def get_transmittance(self, xx, yy, λ):
        # --- Phase computation ---
        phase_x = bd.ones_like(xx)
        # fold into [0,px)
        xmod = (xx % self.px + self.px) % self.px
        for t_frac in self.tx:
            t = t_frac * self.px
            mask = xmod >= t
            # flip sign where xmod>=t
            phase_x = bd.where(mask, -phase_x, phase_x)

        phase_y = bd.ones_like(xx)
        ymod = (yy % self.py + self.py) % self.py
        for t_frac in self.ty:
            t = t_frac * self.py
            mask = ymod >= t
            phase_y = bd.where(mask, -phase_y, phase_y)

        # any region where phase_x*phase_y < 0 gets π shift
        phase2d = bd.where((phase_x * phase_y) < 0,
                            bd.pi * bd.ones_like(xx),
                            bd.zeros_like(xx))

        # --- Amplitude computation ---
        regions_x = bd.zeros_like(xx)
        for t_frac in self.tx:
            t = t_frac * self.px
            regions_x = regions_x + bd.where(xmod >= t,
                                             bd.ones_like(xx),
                                             bd.zeros_like(xx))

        regions_y = bd.zeros_like(xx)
        for t_frac in self.ty:
            t = t_frac * self.py
            regions_y = regions_y + bd.where(ymod >= t,
                                             bd.ones_like(xx),
                                             bd.zeros_like(xx))

        # XOR of parity gives checkerboard; OR for stripes—here we use XOR
        odd = bd.not_equal(regions_x % 2, regions_y % 2)

        # amplitude = 1 everywhere, minus absorbance where odd==True
        amplitude = bd.where(odd,
                             bd.ones_like(xx) - self.absorbance,
                             bd.ones_like(xx))

        # combine amplitude & phase
        t = amplitude * bd.exp(1j * phase2d)

        # --- aperture crop to (width×height) around (x0,y0) ---
        in_aperture = ((xx >= (self.x0 - self.width/2)) &
                       (xx <  (self.x0 + self.width/2)) &
                       (yy >= (self.y0 - self.height/2)) &
                       (yy <  (self.y0 + self.height/2)))
        t = t * bd.where(in_aperture,
                         bd.ones_like(xx),
                         bd.zeros_like(xx))

        return t

class DammannGratingEven(DOE):
    def __init__(self, tx, ty, px, py,
                 width, height,
                 x0=0, y0=0,
                 absorbance=0):
        """
        Creates a 2D even Dammann grating (amplitude + phase) at (x0,y0),
        with transition arrays tx, ty (normalized 0~1), periods px, py,
        and an absorbance (0–1) for the amplitude modulation.
        The grating is cropped to a rectangle of size (width×height).
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.tx = np.sort(np.array(tx))  # sorted transition fractions
        self.ty = np.sort(np.array(ty))
        self.px = px
        self.py = py

        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0

        self.absorbance = absorbance

    def get_transmittance(self, xx, yy, λ):
        # --- Phase computation ---
        phase_x = bd.ones_like(xx)
        # Fold and center coordinates in [-px/2, px/2]
        xmod = ((xx - self.x0 + self.px / 2) % self.px) - self.px / 2
        for t_frac in self.tx:
            t = t_frac * self.px
            mask = bd.abs(xmod) >= t
            phase_x = bd.where(mask, -phase_x, phase_x)

        phase_y = bd.ones_like(xx)
        ymod = ((yy - self.y0 + self.py / 2) % self.py) - self.py / 2
        for t_frac in self.ty:
            t = t_frac * self.py
            mask = bd.abs(ymod) >= t
            phase_y = bd.where(mask, -phase_y, phase_y)

        # Region where phase_x * phase_y < 0 → π phase shift
        phase2d = bd.where((phase_x * phase_y) < 0,
                           bd.pi * bd.ones_like(xx),
                           bd.zeros_like(xx))

        # --- Amplitude computation ---
        regions_x = bd.zeros_like(xx)
        for t_frac in self.tx:
            t = t_frac * self.px
            regions_x += bd.where(bd.abs(xmod) >= t,
                                  bd.ones_like(xx),
                                  bd.zeros_like(xx))

        regions_y = bd.zeros_like(xx)
        for t_frac in self.ty:
            t = t_frac * self.py
            regions_y += bd.where(bd.abs(ymod) >= t,
                                  bd.ones_like(xx),
                                  bd.zeros_like(xx))

        # Even checkerboard: both x and y have even or both odd → light; else → absorb
        even = bd.equal(regions_x % 2, regions_y % 2)

        amplitude = bd.where(even,
                             bd.ones_like(xx),
                             bd.ones_like(xx) - self.absorbance)

        # Combine amplitude and phase
        t = amplitude * bd.exp(1j * phase2d)

        # --- Aperture crop ---
        in_aperture = ((xx >= (self.x0 - self.width/2)) &
                       (xx <  (self.x0 + self.width/2)) &
                       (yy >= (self.y0 - self.height/2)) &
                       (yy <  (self.y0 + self.height/2)))
        t = t * bd.where(in_aperture,
                         bd.ones_like(xx),
                         bd.zeros_like(xx))

        return t




class HoneycombDammannGrating(DOE):
    """
    2D honeycomb (hexagonal) Dammann grating:
    - Phase flips along three 60-degree axes to create a hexagonal tiling of binary phase.
    - Pure-phase (0, π) pattern with optional uniform attenuation.
    - Cropped to a rectangular aperture.

    Parameters:
        t1, t2, t3 : sequences of float
            Transition fractions (within [0,0.5]) for each of the three lattice directions.
        p : float
            Lattice period (distance between repeating hexagon centers).
        width, height : float
            Aperture dimensions.
        x0, y0 : float, optional
            Center offset of the aperture.
        absorbance : float, optional
            Uniform amplitude attenuation (0 = no loss).
    """
    def __init__(self, t1, t2, t3, p,
                 width, height,
                 x0=0.0, y0=0.0,
                 absorbance=0.0):
        self.t1 = np.sort(np.array(t1))
        self.t2 = np.sort(np.array(t2))
        self.t3 = np.sort(np.array(t3))
        self.p = float(p)

        self.width = float(width)
        self.height = float(height)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.absorbance = float(absorbance)

        # Unit vectors for hexagonal lattice axes
        self.e1 = np.array([1.0, 0.0])
        self.e2 = np.array([0.5, np.sqrt(3)/2])
        self.e3 = np.array([-0.5, np.sqrt(3)/2])

    def get_transmittance(self, xx, yy, λ=None):
        # lazy import backend
        from ..util.backend_functions import backend as bd

        # --- Projection and folding for each direction ---
        def folded_fraction(coord_x, coord_y, e_dir):
            # raw projection onto e_dir
            raw = (coord_x - self.x0) * e_dir[0] + (coord_y - self.y0) * e_dir[1]
            # mod into [0, p)
            r = (raw % self.p + self.p) % self.p
            # fold into half-period
            return bd.where(r > self.p/2, self.p - r, r) / self.p

        # Fractions along three axes
        f1 = folded_fraction(xx, yy, self.e1)
        f2 = folded_fraction(xx, yy, self.e2)
        f3 = folded_fraction(xx, yy, self.e3)

        # Build 1D ±1 phase masks
        mask1 = bd.ones_like(xx)
        for t in self.t1:
            mask1 = bd.where(f1 >= t, -mask1, mask1)
        mask2 = bd.ones_like(xx)
        for t in self.t2:
            mask2 = bd.where(f2 >= t, -mask2, mask2)
        mask3 = bd.ones_like(xx)
        for t in self.t3:
            mask3 = bd.where(f3 >= t, -mask3, mask3)

        # Combine three directions: product<0 => π-phase
        phase_sign = mask1 * mask2 * mask3
        phase = bd.where(phase_sign < 0,
                         bd.pi * bd.ones_like(xx),
                         bd.zeros_like(xx))

        # Uniform amplitude
        amplitude = bd.ones_like(xx) * (1.0 - self.absorbance)

        # Complex transmittance
        t = amplitude * bd.exp(1j * phase)

        # Crop to rectangular aperture
        xmin = self.x0 - self.width/2
        xmax = self.x0 + self.width/2
        ymin = self.y0 - self.height/2
        ymax = self.y0 + self.height/2
        in_ap = ((xx >= xmin) & (xx < xmax) &
                 (yy >= ymin) & (yy < ymax))
        t = t * bd.where(in_ap,
                         bd.ones_like(t),
                         bd.zeros_like(t))
        return t
