"""
Technical Paper:
Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang, "Curriculum learning for ab initio deep learned refractive optics," ArXiv preprint (2023)

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from authors).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.
"""
import math
import torch
import numpy as np

DEFAULT_WAVE = 589.3
WAVE_RGB = [656.3, 589.3, 486.1]
WAVE_RGB2 = [630, 532, 465] # real RGB wavelngth
MINT = 1e-5
MAXT = 1e5
DEPTH = -20000
EPSILON = 1e-6  # replace 0 with EPSILON in some cases
GEO_SPP = 512   # spp for geometric optics calculation
NEWTON_STEP_BOUND = 1   # Maximum step length in one Newton iteration

MATERIAL_TABLE = { 
    # [nD, Abbe number]
    "vacuum":       [1.,       math.inf],
    "air":          [1.,       math.inf],
    "occluder":     [1.,       math.inf],
    "f2":           [1.620,    36.37],
    "f5":           [1.6034,   38.03],
    "bk1":          [1.5101,   63.465],
    "bk7":          [1.5168,   64.17],
    
    # https://shop.schott.com/advanced_optics/
    "bk10":         [1.49780,  66.954],
    "kzfs1":        [1.6131,   44.339],
    "laf20":        [1.6825,   48.201],
    "lafn7":        [1.7495,   34.951],
    "n-baf10":      [1.67003,  47.11],
    "n-bk7":        [1.51680,  64.17],
    "n-lak34":      [1.75500,  52.30],
    "n-pk51":       [1.53100,  56.00],
    "n-pk52":       [1.49700,  81.63],
    "n-balf4":      [1.57960,  53.86],
    "n-ssk2":       [1.62229,  53.27],
    "n-sf57":       [1.84666,  23.78],
    "sf11":         [1.87450,  25.68],

    # plastic for cellphone
    # from paper: Analysis of the dispersion of optical plastic materials
    "coc":          [1.5337,   56.22],
    "pmma":         [1.4918,   57.44],
    "ps":           [1.5904,   30.87],
    "pc":           [1.5855,   29.91],
    "okp4ht":       [1.6328,   23.34],
    "okp4":         [1.6328,   23.34],
    
    "apl5014cl":    [1.5445,   55.987],
    "d-k59":        [1.5176,   63.500],

    # SUMITA.AGF
    "sk1":          [1.61020,  56.504],
    "sk16":         [1.62040,  60.306],
    "sk1":          [1.61030,  56.712],
    "sk16":         [1.62040,  60.324],
    "ssk4":         [1.61770,  55.116],

    # https://www.pgo-online.com/intl/B270.html
    "b270":         [1.52290,  58.50],
    
    # https://refractiveindex.info, nd at 589.3 [nm]
    "s-nph1":       [1.8078,   22.76], 
    "d-k59":        [1.5175,   63.50],
    
    "flint":        [1.6200,   36.37],
    "pmma":         [1.491756, 58.00],
    "polycarb":     [1.58547,  29.91],
    "polystyr":     [1.59048,  30.87]
}


SELLMEIER_TABLE = {
    "vacuum":       [0., 0., 0., 0., 0., 0.],
    "air":          [0., 0., 0., 0., 0., 0.],
    "occluder":     [0., 0., 0., 0., 0., 0.],
    "f2":           [1.3453, 9.9774e-3, 2.0907e-1, 4.7045e-2, 9.3736e-1, 1.1188e2],
    "f5":           [1.3104, 9.5863e-3, 1.9603e-1, 4.5762e-2, 9.6612e-1, 1.1501e2],
    "bk1":          [1.0425, 6.1656e-3, 2.0838e-1, 2.1215e-2, 9.8014e-1, 1.0906e2],
    "bk7":          [1.0396, 6.0006e-3, 2.3179e-1, 2.0017e-2, 1.0104,    1.0356e2],
    "sf11":         [1.7385, 1.3607e-2, 3.1117e-1, 6.1596e-2, 1.1749,    1.2192e2],
    
    # https://shop.schott.com/advanced_optics/
    "kzfs1":        [1.3661, 8.7316e-3, 1.8204e-1, 3.8983e-2, 8.6431e-1, 6.2939e1],
    "laf20":        [1.6510, 9.7050e-3, 1.1847e-1, 4.2892e-2, 1.1154, 1.1405e2],
    "lafn7":        [1.6684, 1.0316e-2, 2.9851e-1, 4.6922e-2, 1.0774, 8.2508e1],
    "n-bk7":        [1.0396, 6.0006e-3, 2.3179e-1, 2.0017e-2, 1.0104, 1.0356e2],
    "n-lak34":      [1.2666, 5.8928e-3, 6.6592e-1, 1.9751e-2, 1.1247, 78.889],
    "n-pk51":       [1.1516, 5.8556e-3, 1.5323e-1, 1.9407e-2, 7.8562e-1, 140.537],
    "n-pk52":       [1.0081, 5.0197e-3, 2.0943e-1, 1.6248e-2, 7.8169e-1, 1.5239e2],
    "n-balf4":      [1.3100, 7.9659e-3, 1.4208e-1, 3.3067e-2, 9.6493e-1, 1.0919e2],

    # SUMITA.AGF
    "sk16":         [1.3431, 7.0468e-3, 2.4114e-1, 2.2900e-2, 9.9432e-1, 9.2751e1],

    # https://www.pgo-online.com/intl/B270.html
    
    # https://refractiveindex.info, nd at 589.3 [nm]
    "d-k59":        [1.1209, 6.5791e-3, 1.5269e-1, 2.3572e-2, 1.0750, 1.0631e2]
}


SCHOTT_TABLE = {
    "coc":          [2.28449,  1.02952e-2, 3.73494e-2, -9.28410e-3, 1.73290e-3, -1.15203e-4],
    "pmma":         [2.18646, -2.44753e-4, 1.41558e-2, -4.43298e-4, 7.76643e-5, -2.99364e-6],
    "ps":           [2.44598,  2.21429e-5, 2.72989e-2,  3.01211e-4, 8.88934e-5, -1.75708e-6],
    'polystyr':     [2.44598,  2.21429e-5, 2.72989e-2,  3.01211e-4, 8.88934e-5, -1.75708e-6],
    "pc":           [2.42839, -3.86117e-5, 2.87574e-2, -1.97897e-4, 1.48359e-4,  1.38652e-6],
    'polycarb':     [2.42839, -3.86117e-5, 2.87574e-2, -1.97897e-4, 1.48359e-4,  1.38652e-6],
    "okp4ht":       [2.55219,  6.51282e-5, 3.57452e-2,  8.49831e-4, 8.47777e-5,  1.58990e-5],
    "okp4":         [2.49230, -1.46713e-3, 3.04056e-2, -2.31960e-4, 3.62928e-4, -1.89103e-5]
}


GLASS_NAME = {
    "coc":          'COC',
    "pmma":         'PMMA',
    "ps":           'POLYSTYR',
    'polystyr':     'POLYSTYR',
    "pc":           'POLYCARB',
    "polycarb":     'POLYCARB',
    "okp4":         'OKP4',
    "okp4ht":       'OKP4HT'
}



# ----------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------
class Ray():
    def __init__(self, o, d, wavelength=DEFAULT_WAVE, weight=1., normalized=True, ra=None, en=None, obliq=None, opl=None, coherent=False, device=torch.device('cuda:0')):
        """ Ray class. A group of rays with the same wavelength.

        Args:
            o (Tensor): ray position. shape [..., 3]
            d (Tensor): normalized ray direction. shape [..., 3]
            wavelength (float, optional): wavelength. Defaults to DEFAULT_WAVE.
            weight (float, optional): Monte Carlo sampling weight. Defaults to 1..
            normalized (bool, optional): If the vector 'd' is normalized. Defaults to True.
            ra (Tensor, optional): Validity. Defaults to None.
            en (Tensor, optional): Spherical wave energy decay. Defaults to None.
            obliq (Tensor, optional): Obliquity energy decay, now only used to record refractive angle. Defaults to None.
            opl (Tensor, optional): Optical path length, Now used as the phase term. Defaults to None.
            coherent (bool, optional): If the ray is coherent. Defaults to False.
            device (torch.device, optional): Defaults to torch.device('cuda:0').
        """
        self.o = o if torch.is_tensor(o) else torch.tensor(o)
        self.d = d if torch.is_tensor(d) else torch.tensor(d)
        self.coherent = coherent    # if coherent ray, we will record the phase term

        self.ra = ra if ra is not None else torch.full(o.shape[:-1], 1., dtype=torch.float32)
        self.en = en if en is not None else torch.full(o.shape[:-1], 1., dtype=torch.float32)
        self.obliq = obliq if obliq is not None else torch.full(o.shape[:-1], 1., dtype=torch.float32)
        self.opl = opl if opl is not None else torch.full(o.shape[:-1], 0., dtype=torch.float32)
        self.phase = self.opl
        
        if isinstance(weight, float):
            self.weight = torch.full(o.shape[:-1], weight, dtype=torch.float32)
        elif torch.is_tensor(weight):
            self.weight = weight
        else:   # ndarray
            self.weight = torch.tensor(weight, dtype=torch.float32)
        
        self.wavelength = wavelength if wavelength > 100 else wavelength * 1e3 # store wavelength to [nm].
        self.to(device)

        if not normalized:
            self.d = self.d / torch.linalg.vector_norm(self.d, ord=2, dim=-1, keepdim=True)

        # convert to float32
        self.o = self.o.type(torch.float32)
        self.d = self.d.type(torch.float32)

    
    def to(self, device=torch.device('cuda')):
        """ Move all variables to target device.

        Args:
            device (cpu or cuda, optional): target device. Defaults to torch.device('cpu').
        """
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))

        self.device = device
        return self


    def prop_to(self, z, n=1):
        """ Ray propagates to a given depth. 
            
            Old implementation, use self.propagate_to() instead.
        """
        return self.propagate_to(z, n)


    def propagate_to(self, z, n=1):
        """ Ray propagates to a given depth.

            Args:
                z (float): depth.
                n (float, optional): refractive index. Defaults to 1.
        """
        t = (z - self.o[..., 2]) / self.d[..., 2]
        self.o = self.o + self.d * t[..., None]
        
        return self


    def project_to(self, z):
        """ Calculate the intersection points of ray with plane z.

            Return:
                p: shape of [..., 2].
        """
        t = (z - self.o[...,2]) / self.d[...,2]
        p = self.o[...,0:2] + self.d[...,0:2] * t[...,None]
        return p


    def clone(self, device=None):
        """ Clone a Ray object.
            
            Can spercify which device we want to clone. Sometimes we want to store all rays in CPU, and when using it, we move it to GPU.
        """
        o = self.o.clone().detach()
        d = self.d.clone().detach()
        wv = self.wavelength
        weight = self.weight.clone().detach()
        ra = self.ra.clone().detach()
        en = self.en.clone().detach()
        obliq = self.obliq.clone().detach()
        opl = self.opl.clone().detach()
        coherent = self.coherent

        if device is None:
            return Ray(o=o, d=d, wavelength=wv, weight=weight, ra=ra, en=en, obliq=obliq, opl=opl, coherent=coherent, device=self.device)
        else:
            return Ray(o=o, d=d, wavelength=wv, weight=weight, ra=ra, en=en, obliq=obliq, opl=opl, coherent=coherent, device=device)


    def index_clone(self, bl, tr):
        """ Clone a patch of rays. This function is used when we want to render images patch by patch.

            Args:
                bl (tuple): bottom left corner of the patch.
                tr (tuple): top right corner of the patch.
        """
        low_i, low_j = bl
        up_i, up_j = tr
        o_patch = self.o[:, low_i:up_i, low_j:up_j, :].detach().clone()
        d_patch = self.d[:, low_i:up_i, low_j:up_j, :].detach().clone()
        w_patch = self.weight[:, low_i:up_i, low_j:up_j].detach().clone()
        ra_patch = self.ra[:, low_i:up_i, low_j:up_j].detach().clone()
        en_patch = self.en[:, low_i:up_i, low_j:up_j].detach().clone()
        obliq_patch = self.obliq[:, low_i:up_i, low_j:up_j].detach().clone()
        opl_patch = self.opl[:, low_i:up_i, low_j:up_j].detach().clone()
        wavelength = self.wavelength
        coherent = self.coherent
        sub_ray = Ray(o=o_patch, d=d_patch, weight=w_patch, wavelength=wavelength, ra=ra_patch, en=en_patch, obliq=obliq_patch, opl=opl_patch, coherent=coherent, device=self.device)
        return sub_ray


class Material():
    def __init__(self, name=None):
        # We can find commercial glasses here:
        # https://www.schott.com/en-dk/interactive-abbe-diagram
        # https://refractiveindex.info/ 
        
        self.name = 'vacuum' if name is None else name.lower()
        self.A, self.B = self._lookup_material()
        
        if self.name in SELLMEIER_TABLE:
            self.dispersion = 'sellmeier'
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = SELLMEIER_TABLE[self.name]
            self.glassname = self.name
        elif self.name in SCHOTT_TABLE:
            self.dispersion = 'schott'
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = SCHOTT_TABLE[self.name]
            self.glassname = GLASS_NAME[self.name]
        else:
            self.dispersion = 'naive'


    def ior(self, wavelength):
        """ Compute the refractive index at given wavelength. 
            Reference: Zemax user manual.
            
            Args:
                wavelength (float): wavelength in [nm].
        """
        wv = wavelength if wavelength < 100 else wavelength * 1e-3 # Convert to [um]
        
        # Compute refraction index
        if self.dispersion == 'sellmeier':
            n2 = 1 + self.k1 * wv**2 / (wv**2 - self.l1) + self.k2 * wv**2 / (wv**2 - self.l2) + self.k3 * wv**2 / (wv**2 - self.l3)
            n = np.sqrt(n2)
        elif self.dispersion == 'schott':
            # Write dispersion function seperately will introduce errors 
            # High precision computation (by MATLAB):
            ws = wv**2
            n2 = self.a0 + self.a1*ws + (self.a2 + (self.a3 + (self.a4 + self.a5/ws)/ws)/ws)/ws
            n = np.sqrt(n2)
        elif self.dispersion == 'naive':
            # Use Cauchy function
            n = self.A + self.B / (wv * 1e3)**2

        return n


    def load_sellmeier_param(self, params=None):
        """ Manually set sellmeier parameters k1, l1, k2, l2, k3, l3.
            
            This function is used when we want to use a custom material.
        """
        if params is None:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = 0, 0, 0, 0, 0, 0
        else:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = params


    @staticmethod
    def nV_to_AB(n, V):
        """ Convert (n ,V) paramters to (A, B) parameters to find the material.
        """
        def ivs(a): return 1./a**2
        lambdas = [656.3, 589.3, 486.1]
        B = (n - 1) / V / ( ivs(lambdas[2]) - ivs(lambdas[0]) )
        A = n - B * ivs(lambdas[1])
        return A, B


    def _lookup_material(self):
        """ Find (A, B) parameters of the material. (A, B) parameters are used to calculate the refractive index in the old implementation (by Cauchy's equation).
        """
        out = MATERIAL_TABLE.get(self.name)
        if isinstance(out, list):
            n, V = out
        elif out is None:
            # try parsing input as a n/V pair
            tmp = self.name.split('/')
            n, V = float(tmp[0]), float(tmp[1])

        self.n = n
        self.V = V
        return self.nV_to_AB(n, V)
