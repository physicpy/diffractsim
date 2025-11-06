import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE



class BinaryGrating(DOE):
    def __init__(self, period, width, height, x0=0, y0=0, rotate90=False):
        """
        Creates a binary (amplitude) rectangular grating at the point (x0, y0) 
        with width `width` and height `height`. 
        
        If rotate90=True, the grating pattern is rotated by 90 degrees.
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.period = period
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height
        self.rotate90 = rotate90  # <--- new variable

    def get_transmittance(self, xx, yy, λ):
        # Choose direction of periodic modulation
        if self.rotate90:
            coord = yy  # grating lines along x-axis (rotated)
        else:
            coord = xx  # grating lines along y-axis (default)

        # Binary grating: 1 for half the period, 0 for the other half
        t = bd.sign((coord) % self.period - self.period / 2)
        t = bd.select([t == 0, t == 1, t == -1],
                      [bd.ones_like(t), bd.ones_like(t), bd.zeros_like(t)])

        # Apply aperture window
        in_window = ((xx >= (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)) &
                     (yy >= (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))
        t = t * bd.where(in_window, bd.ones_like(xx), bd.zeros_like(xx))

        return t




class PhaseGrating(DOE):
    def __init__(self, period, width, height, x0 = 0, y0 = 0):
        """
        Creates a phase grating at the point (x0, y0) with width width and height height
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.period = period
        self.x0 = x0
        self.y0 = y0

        self.width = width
        self.height = height

    def get_transmittance(self, xx, yy, λ):


        t = bd.where((((xx > (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)))
                        & ((yy > (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
                        bd.ones_like(xx), bd.zeros_like(xx))

        phase_shift = xx/self.period
        return t*bd.exp(1j*2*bd.pi*phase_shift)




class BinaryPhaseGrating(DOE):
    def __init__(self, period, width, height, x0=0, y0=0):
        """
        Creates a binary *phase* rectangular grating centered at (x0, y0) with given width and height.
        Phase values alternate between 0 and pi across each period along x.
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.period = period
        self.x0 = x0
        self.y0 = y0

        self.width = width
        self.height = height

    def get_transmittance(self, xx, yy, λ):
        # rectangular support mask (1 inside the rectangle, 0 outside)
        rect_mask = bd.where(
            (((xx > (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)))
             & ((yy > (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
            bd.ones_like(xx),
            bd.zeros_like(xx)
        )

        # create a binary pattern across the period: 1 for one half-period, 0 for the other
        s = bd.sign((xx) % (self.period) - self.period / 2)
        # map sign outputs to binary: positive / zero -> 1, negative -> 0
        binary_half = bd.select(
            [s == 0, s == 1, s == -1],
            [bd.ones_like(s), bd.ones_like(s), bd.zeros_like(s)]
        )

        # phase is 0 or pi depending on the binary pattern
        phase = binary_half * bd.pi  # values are 0 or pi

        # complex transmittance: exp(i * phase) inside rectangle, zero outside
        return rect_mask * bd.exp(1j * phase)



class BinaryVLSGrating(DOE):
    def __init__(self, N, period, width, height, x0=0, y0=0):
        """
        Creates a binary amplitude Variable-Line-Spacing (VLS) grating centered at (x0, y0).
        
        Parameters
        ----------
        N : int
            Number of grating lines.
        period : float
            Approximate period scaling factor; defines the total span of the grating in x.
        width : float
            Total grating width (in x-direction).
        height : float
            Grating height (in y-direction).
        x0, y0 : float
            Center coordinates of the grating.
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.N = N
        self.period = period
        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0

        # Precompute edge and center positions (normalized 0–1, then scaled to width)
        i = np.arange(0, N + 1)
        edges = (i / N) ** 2  # quadratic edge positions
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.edges = edges * width - width / 2 + x0
        self.centers = centers * width - width / 2 + x0

    def get_transmittance(self, xx, yy, λ):
        # rectangular mask for finite aperture
        rect_mask = bd.where(
            (((xx > (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2))) &
             ((yy > (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
            bd.ones_like(xx),
            bd.zeros_like(xx)
        )

        # start with zeros
        t = bd.zeros_like(xx)

        # iterate over each line region defined by variable spacing
        for n in range(self.N):
            x_left = self.edges[n]
            x_right = self.edges[n + 1]

            # binary amplitude: 1 for even lines, 0 for odd (or invert as desired)
            line_value = 1 if n % 2 == 0 else 0

            t += bd.where((xx >= x_left) & (xx < x_right),
                          line_value * bd.ones_like(xx),
                          bd.zeros_like(xx))

        return t * rect_mask



class BinaryPhaseVLSGrating(DOE):
    def __init__(self, N, width, height, x0=0, y0=0, phase_high=np.pi, invert=False, mirror=False):
        """
        Binary variable-line-spacing (VLS) phase grating (quadratic spacing).
        Each stripe alternates between phase 0 and phase_high (default π).
        Supports optional inversion and mirroring along x.

        Parameters
        ----------
        N : int
            Number of grating lines (number of stripes).
        width : float
            Total grating width (x span).
        height : float
            Grating height (y span).
        x0, y0 : float
            Center position of the grating.
        phase_high : float
            Phase value for the 'high' stripes (default: np.pi).
        invert : bool
            If True, invert stripe assignment (odd/even phase swap).
        mirror : bool
            If True, mirror the grating pattern along the x-direction.
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.N = int(N)
        self.width = width
        self.height = height
        self.x0 = x0
        self.y0 = y0
        self.phase_high = float(phase_high)
        self.invert = bool(invert)
        self.mirror = bool(mirror)

        # Compute normalized quadratic edges (0..1), optionally mirrored
        i = np.arange(0, self.N + 1)
        edges_norm = (i / self.N) ** 2  # quadratic spacing of transition points
        if self.mirror:
            edges_norm = 1.0 - edges_norm[::-1]  # flip direction (mirror)

        self.edges = edges_norm * self.width + (self.x0 - self.width / 2)

        # Centers (for possible diagnostics or plotting)
        centers_norm = 0.5 * (edges_norm[:-1] + edges_norm[1:])
        self.centers = centers_norm * self.width + (self.x0 - self.width / 2)

    def get_transmittance(self, xx, yy, λ):
        """
        Returns complex transmittance: exp(i * phase) inside the rectangle,
        with phase alternating between 0 and phase_high according to VLS stripes.
        """
        # rectangular support mask (1 inside rectangle, 0 outside)
        rect_mask = bd.where(
            (((xx > (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2))) &
             ((yy > (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
            bd.ones_like(xx),
            bd.zeros_like(xx)
        )

        # build binary stripe map t (0 or 1)
        t = bd.zeros_like(xx)
        for n in range(self.N):
            x_left = float(self.edges[n])
            x_right = float(self.edges[n + 1])

            # choose which stripe index carries the 'high' phase
            stripe_is_high = ((n % 2) == 0)  # even stripes high by default
            if self.invert:
                stripe_is_high = not stripe_is_high

            line_value = 1.0 if stripe_is_high else 0.0

            t = t + bd.where((xx >= x_left) & (xx < x_right),
                             line_value * bd.ones_like(xx),
                             bd.zeros_like(xx))

        # phase: 0 or phase_high
        phase = t * self.phase_high

        # complex transmittance inside rectangle; outside -> 0
        return rect_mask * bd.exp(1j * phase)