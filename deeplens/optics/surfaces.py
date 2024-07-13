""" Geometric surfaces for ray tracing.
"""
import torch
import math
import numpy as np
import torch.nn.functional as nnF
import matplotlib.pyplot as plt

from .basics import *
from .materials import Material
from ..utils import *

class Surface(DeepObj):
    def __init__(self, r, d, mat2, is_square=False, device=DEVICE):
        super(Surface, self).__init__()
        self.d = d if torch.is_tensor(d) else torch.Tensor([d])
        self.d_perturb = 0.0

        self.r = float(r)   # r is not differentiable
        self.r_perturb = 0.0
        self.is_square = is_square
        if is_square:
            self.h = r * math.sqrt(2)
            self.w = r * math.sqrt(2)
        
        # self.mat1 = Material('air')
        self.mat2 = Material(mat2)        

        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 10e-6 # in [mm], here is 10 [nm] 
        self.NEWTONS_TOLERANCE_LOOSE = 50e-6 # in [mm], here is 50 [nm] 
        self.NEWTONS_STEP_BOUND = 5 # [mm], maximum iteration step in Newton's iteration
        self.APERTURE_SAMPLING = 257

        self.to(device)

    # ==============================
    # Intersection and Refraction
    # ==============================
    def ray_reaction(self, ray, n1=None, n2=None):
        """ Compute output ray after intersection and refraction with a surface.
        """
        if n1 is None and n2 is None:
            # Determine ray direction and refractive index
            wvln = ray.wvln
            forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
            raise Exception('In the future, we need to specify the refractive index of the two media.')
            if forward:
                n1 = self.mat1.ior(wvln)
                n2 = self.mat2.ior(wvln)
            else:
                n1 = self.mat2.ior(wvln)
                n2 = self.mat1.ior(wvln)

        # Intersection
        ray = self.intersect(ray, n1)

        # Refraction
        ray = self.refract(ray, n1 / n2)

        return ray
    
    def intersect(self, ray, n):
        """ Solve ray-surface intersection and update ray position and opl.

        Args:
            n (float, optional): refractive index. Defaults to 1.0.
        """
        # Solve intersection time t by Newton's method
        t, valid = self.newtons_method(ray)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            assert t.min() < 100, 'Precision problem caused by long propagation distance.'
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def newtons_method(self, ray):
        """ Solve intersection by Newton's method. 
        
            This function will only update valid rays.
        """
        d_surf = self.d + self.d_perturb

        # 1. inital guess of t
        t0 = (d_surf - ray.o[...,2]) / ray.d[...,2]   # if the shape of aspheric surface is strange, will hit the back surface region instead 

        # 2. use Newton's method to update t to find the intersection points (non-differentiable)
        with torch.no_grad():
            it = 0
            t = t0  # initial guess of t
            ft = MAXT * torch.ones_like(ray.o[...,2])
            while (torch.abs(ft) > self.NEWTONS_TOLERANCE_LOOSE).any() and (it < self.NEWTONS_MAXITER):
                it += 1

                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[...,0], new_o[...,1]
                valid = self.valid(new_x, new_y) & (ray.ra>0)
                
                ft = self.sag(new_x, new_y, valid) + d_surf - new_o[...,2]
                dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
                dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y)
                dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
                t = t - torch.clamp(ft / (dfdt+1e-9), - self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

            t1 = t - t0

        # 3. do one more Newton iteration to gain gradients
        t = t0 + t1

        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[...,0], new_o[...,1]
        valid = self.valid(new_x, new_y) & (ray.ra > 0)
        
        ft = self.sag(new_x, new_y, valid) + d_surf - new_o[...,2]
        dxdt, dydt, dzdt = ray.d[...,0], ray.d[...,1], ray.d[...,2]
        dfdx, dfdy, dfdz = self.dfdxyz(new_x, new_y)
        dfdt = dfdx * dxdt + dfdy * dydt + dfdz * dzdt
        t = t - torch.clamp(ft / (dfdt+1e-9), - self.NEWTONS_STEP_BOUND, self.NEWTONS_STEP_BOUND)

        # determine valid rays
        with torch.no_grad():
            new_x, new_y = new_o[...,0], new_o[...,1]
            valid = self.valid_within_boundary(new_x, new_y) & (ray.ra > 0)
            ft = self.sag(new_x, new_y, valid) + d_surf - new_o[...,2]
            valid = valid & (torch.abs(ft.detach()) < self.NEWTONS_TOLERANCE_TIGHT) & (t > 0)   # points valid & points accurate & donot go back
        
        return t, valid


    def refract(self, ray, eta):
        """ Calculate refractive ray according to Snell's law.
        
            Snell's law (surface normal n defined along the positive z axis):
                https://physics.stackexchange.com/a/436252/104805
                https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
                We follow the first link and normal vector should have the same direction with incident ray(veci), but by default it points to left. We use the second link to check.

                veci: incident ray
                vect: refractive ray
                eta: relevant refraction coefficient, eta = eta_i/eta_t
        """
        # Compute normal vectors
        n = self.normal(ray)
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0
        if forward:
            n = - n

        # Compute refraction according to Snell's law
        cosi = torch.sum(ray.d * n, axis=-1)   # n * i

        # TIR
        valid = (eta**2 * (1 - cosi**2) < 1) & (ray.ra > 0)

        sr = torch.sqrt(1 - eta**2 * (1 - cosi.unsqueeze(-1)**2) * valid.unsqueeze(-1) + EPSILON)  # square root
        
        # First term: vertical. Second term: parallel. Already normalized if both n and ray.d are normalized. 
        new_d = sr * n + eta * (ray.d - cosi.unsqueeze(-1) * n)
        new_d[~valid] = ray.d[~valid]

        new_obliq = torch.sum(new_d * ray.d, axis=-1)
        new_obliq[~valid] = ray.obliq[~valid]

        # Update valid rays
        ray.d = new_d
        ray.obliq = new_obliq
        ray.ra = ray.ra * valid

        return ray


    def normal(self, ray):
        """ Calculate normal vector of the surface.

            Normal vector points to the left by default.
        """
        x, y, z = ray.o[...,0], ray.o[...,1], ray.o[...,2]
        nx, ny, nz = self.dfdxyz(x, y)
        n = torch.stack((nx, ny, nz), axis = -1)
        n = nnF.normalize(n, p = 2, dim = -1)

        return n
    
    # =================================================================================
    # Calculation-related methods
    # =================================================================================
    def sag(self, x, y, valid=None):
        """ Calculate sag (z) of the surface. z = f(x, y)

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        if valid is None:
            valid = self.valid(x, y)
        
        x, y = x * valid, y * valid
        return self.g(x, y)

    def dfdxyz(self, x, y, valid=None):
        """ Compute derivatives of surface function. Surface function: f(x, y, z): z - g(x, y) = 0

            This function only works for surfaces which can be written as z = g(x, y). For implicit surfaces, we need to compute derivatives (df/dx, df/dy, df/dz).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate

        Return:
            dfdx (tensor): df / dx
            dfdy (tensor): df / dy
            dfdz (tensor): df / dz
        """
        if valid is None:
            valid = self.valid(x, y)
        
        x, y = x * valid, y * valid
        dx, dy = self.dgd(x, y)
        return dx, dy, - torch.ones_like(x)
    
    def g(self, x, y):
        """ Calculate sag (z) of the surface. z = f(x, y)

            Valid term is used to avoid NaN when x, y are super large, which happens in spherical and aspherical surfaces. 
            
            If you want to calculate r = sqrt(x**2, y**2), this will cause another NaN error when calculating dr/dx = x / sqrt(x**2 + y**2). So be careful for this!!!

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            valid (tensor): valid mask

        Return:
            z (tensor): z = sag(x, y)
        """
        raise NotImplementedError()
        
    def dgd(self, x, y):
        """ Compute derivatives of sag to x and y. (dgdx, dgdy) =  (g'x, g'y).

        Args:
            x (tensor): x coordinate
            y (tensor): y coordinate
            
        Return:
            dgdx (tensor): dg / dx
            dgdy (tensor): dg / dy
        """
        raise NotImplementedError()
    
    def is_valid(self, p):
        return (self.sdf_approx(p) < 0.0).bool()

    def valid_within_boundary(self, x, y):
        """ Valid points within the boundary of the surface.
        """
        if self.is_square:
            valid = self.valid(x, y) & (torch.abs(x) <= self.w/2) & (torch.abs(y) <= self.h/2)
        else:
            valid = self.valid(x, y) & ((x**2 + y**2) <= self.r**2)
        
        return valid
    
    def valid(self, x, y):
        """ Valid points NOT considering the boundary of the surface.
        """
        return torch.ones_like(x, dtype=torch.bool)
    
    def surface_sample(self, N=1000):
        """ Sample uniform points on the surface.
        """
        r_max = self.r
        theta = torch.rand(N)*2*np.pi
        r = torch.sqrt(torch.rand(N)*r_max**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, self.d.item())
        o2 = torch.stack((x2,y2,z2), 1).to(self.device)
        return o2
    
    def surface(self, x, y):
        """ Calculate z coordinate of the surface at (x, y) with offset.
            
            This function is used in lens setup plotting.
        """
        x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
        return self.sag(x, y)
    
    def surface_with_offset(self, x, y):
        """ Calculate z coordinate of the surface at (x, y) with offset.
            
            This function is used in lens setup plotting.
        """
        x = x if torch.is_tensor(x) else torch.tensor(x).to(self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y).to(self.device)
        return self.sag(x, y) + self.d
    
    def max_height(self):
        """ Maximum valid height.
        """
        raise NotImplementedError()
    
    # =========================================
    # Optimization-related methods
    # =========================================
    def activate_grad(self, activate=True):
        raise NotImplementedError()
    
    def get_optimizer_params(self, lr):
        raise NotImplementedError()

    def get_optimizer(self, lr):
        params = self.get_optimizer_params(lr)
        return torch.optim.Adam(params)

    def surf_dict(self):
        surf_dict = {
            'type': self.__class__.__name__,
            'r': self.r,
            'd': self.d.item(),
            'is_square': self.is_square,
            # 'mat1': self.mat1.name,
            'mat2': self.mat2.name,
        }

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        raise NotImplementedError()
    
    @torch.no_grad()
    def perturb(self, d_precision=0.0005, r_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors.
        """
        self.r_perturb = self.r.item() * float(np.random.randn() * r_precision)
        self.d_perturb = float(torch.randn() * d_precision)




















class Aperture(Surface):
    def __init__(self, r, d, diffraction=False, device=DEVICE):
        """ Aperture, can be circle or rectangle. 
            For geo optics, it works as a binary mask.
            For wave optics, it works as a diffractive plane.
        """
        Surface.__init__(self, r, d, mat2='air', is_square=False, device=device)
        self.diffraction = diffraction
        self.to(device)

    def ray_reaction(self, ray, n1=1.0, n2=1.0):
        """ Compute output ray after intersection and refraction.

            In each step, first get a guess of new o and d, then compute valid and only update valid rays. 
        """
        # -------------------------------------
        # Intersection
        # ------------------------------------- 
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) <= self.r) & (ray.ra > 0)

        # => Update position
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        # => Update phase
        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        # -------------------------------------
        # Diffraction
        # ------------------------------------- 
        if self.diffraction:
            raise Exception('Unimplemented diffraction method.')
            # We can use Huygens-Fresnel principle to determine the diffractive rays, but how to make this process differentiable???? Using Heisenberg uncertainty principle???
            # Ref: Simulating multiple diffraction in imaging systems using a path integration method
            # Conventional method process the aperture by using the exit pupil + free space propagation, if that we donot need this class.

        return ray
    
    def g(self, x, y):
        """ Compute surface height.
        """
        return torch.zeros_like(x)
    
    def surf_dict(self):
        """ Return a dict of surface.
        """
        surf_dict = {
            'type': 'Aperture',
            'r': self.r,
            'd': self.d.item(),
            'is_square': self.is_square,
            'diffraction': self.diffraction,
            }
        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
"""
        return zmx_str

class Aspheric(Surface):
    """ This class can represent plane, spheric and aspheric surfaces.

        Aspheric surface: https://en.wikipedia.org/wiki/Aspheric_lens.

        Three kinds of surfaces:
            1. flat: always use round 
            2. spheric: 
            3. aspheric: 
    """
    def __init__(self, r, d, c=0., k=0., ai=None, mat2=None, device=DEVICE):
        """ Initialize aspheric surface.

        Args:
            r (float): radius of the surface
            d (tensor): distance from the origin to the surface
            c (tensor): curvature of the surface
            k (tensor): conic constant
            ai (list of tensors): aspheric coefficients
            mat1 (Material): material of the first medium
            mat2 (Material): material of the second medium
            is_square (bool): whether the surface is square
            device (torch.device): device to store the tensor
        """
        Surface.__init__(self, r, d, mat2, is_square=False, device=device)
        self.c = torch.Tensor([c])
        self.k = torch.Tensor([k])
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))
            self.ai_degree = len(ai)
            if self.ai_degree == 4:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[3]])
            elif self.ai_degree == 5:
                self.ai2 = torch.Tensor([ai[0]])
                self.ai4 = torch.Tensor([ai[1]])
                self.ai6 = torch.Tensor([ai[2]])
                self.ai8 = torch.Tensor([ai[3]])
                self.ai10 = torch.Tensor([ai[4]])
            elif self.ai_degree == 6:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
            else:
                for i, a in enumerate(ai):
                    exec(f'self.ai{2*i+2} = torch.Tensor([{a}])')
        else:
            self.ai = None
            self.ai_degree = 0
        
        self.to(device)


    def init(self, ai_degree=6):
        """ Initialize all parameters.
        """
        self.init_c()
        self.init_k()
        self.init_ai(ai_degree=ai_degree)
        self.init_d()


    def init_c(self, c_bound=0.0002):
        """ Initialize lens surface c parameters by small values between [-0.05, 0.05], 
            which means roc should be (-inf, 20) or (20, inf)
        """
        self.c = c_bound * (torch.rand(1) - 0.5).to(self.device)

    def init_ai(self, ai_degree=3, bound=0.0001):
        """ If ai is None, set to random value.
            For different length, create a new initilized value and set original ai.
        """
        old_ai_degree = self.ai_degree
        self.ai_degree = ai_degree
        if old_ai_degree == 0:
            if ai_degree == 4:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
            elif ai_degree == 5:
                self.ai2 = (torch.rand(1, device=self.device)-0.5) * bound * 10
                self.ai4 = (torch.rand(1, device=self.device)-0.5) * bound
                self.ai6 = (torch.rand(1, device=self.device)-0.5) * bound * 0.1
                self.ai8 = (torch.rand(1, device=self.device)-0.5) * bound * 0.01
                self.ai10 = (torch.rand(1, device=self.device)-0.5) * bound* 0.001
            elif ai_degree == 6:
                for i in range(1, self.ai_degree+1):
                    exec(f'self.ai{2 * i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i - 2}')
            else:
                raise Exception('Wrong ai degree')
        else:
            for i in range(old_ai_degree + 1, self.ai_degree + 1):
                exec(f'self.ai{2 * i} = (torch.rand(1, device=self.device)-0.5) * bound * 0.1 ** {i - 2}')

    
    def init_k(self, bound=1):
        """ When k is 0, set to a random value.
        """
        if self.k == 0:
            k = torch.rand(1) * bound
            self.k = k.to(self.device) 


    def init_d(self, bound = 0.1):
        return

    
    def g(self, x, y):
        """ Compute surface height.
        """
        r2 = x**2 + y**2
        total_surface = r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON))

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4
            elif self.ai_degree == 5:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5
            elif self.ai_degree == 6:
                total_surface = total_surface + self.ai2 * r2 + self.ai4 * r2 ** 2 + self.ai6 * r2 ** 3 + self.ai8 * r2 ** 4 + self.ai10 * r2 ** 5 + self.ai12 * r2 ** 6
            elif self.ai_degree == 7:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + self.ai14 * r2) * r2) * r2) * r2) * r2) * r2) * r2
            elif self.ai_degree == 8:
                total_surface = total_surface + (self.ai2 + (self.ai4 + (self.ai6 + (self.ai8 + (self.ai10 + (self.ai12 + (self.ai14 + self.ai16 * r2)* r2) * r2) * r2) * r2) * r2) * r2) * r2
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f'total_surface += self.ai{2*i} * r2 ** {i}')

        return total_surface


    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        r2 = x**2 + y**2
        sf = torch.sqrt(1 - (1 + self.k) * r2 * self.c**2 + EPSILON)
        dsdr2 = (1 + sf + (1 + self.k) * r2 * self.c**2 / 2 / sf) * self.c / (1 + sf)**2

        if self.ai_degree > 0:
            if self.ai_degree == 4:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3
            elif self.ai_degree == 5:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4
            elif self.ai_degree == 6:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4 + 6 * self.ai12 * r2 ** 5
            elif self.ai_degree == 7:
                dsdr2 = dsdr2 + self.ai2 + 2 * self.ai4 * r2 + 3 * self.ai6 * r2 ** 2 + 4 * self.ai8 * r2 ** 3 + 5 * self.ai10 * r2 ** 4 + 6 * self.ai12 * r2 ** 5 + 7 * self.ai14 * r2 ** 6
            elif self.ai_degree == 8:
                dsdr2 = dsdr2 + self.ai2 + (2 * self.ai4 + (3 * self.ai6 + (4 * self.ai8 + (5 * self.ai10 + (6 * self.ai12 + (7 * self.ai14 + 8 * self.ai16 * r2)* r2)* r2)* r2) * r2) * r2) * r2 
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f'dsdr2 += {i} * self.ai{2*i} * r2 ** {i-1}')

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def valid(self, x, y):
        """ Invalid when shape is non-defined.
        """
        if self.k > -1:
            valid = ((x**2 + y**2) < 1 / self.c**2 / (1 + self.k))
        else:
            valid = torch.ones_like(x, dtype=torch.bool)

        return valid

    def max_height(self):
        """ Maximum valid height.
        """
        if self.k > -1:
            max_height = torch.sqrt(1 / (self.k + 1) / (self.c**2)).item() - 0.01
        else:
            max_height = 100

        return max_height

    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-2], decay=0.01):
        """ Get optimizer parameters for different parameters.

        Args:
            lr (list, optional): learning rates for c, d, k, ai. Defaults to [1e-4, 1e-4, 1e-1, 1e-4].
            decay (float, optional): decay rate for ai. Defaults to 0.1.
        """
        if isinstance(lr, float):
            lr = [lr, lr, lr*1e3, lr]

        params = []
        if lr[0] > 0 and self.c != 0:
            self.c.requires_grad_(True)
            params.append({'params': [self.c], 'lr': lr[0]})
        
        if lr[1] > 0:
            self.d.requires_grad_(True)
            params.append({'params': [self.d], 'lr': lr[1]})
        
        if lr[2] > 0 and self.k != 0:
            params.append({'params': [self.k], 'lr': lr[2]})
        
        if lr[3] > 0:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr[3]})
                params.append({'params': [self.ai4], 'lr': lr[3] * decay})
                params.append({'params': [self.ai6], 'lr': lr[3] * decay**2})
                params.append({'params': [self.ai8], 'lr': lr[3] * decay**3})
            elif self.ai_degree == 5:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr[3]})
                params.append({'params': [self.ai4], 'lr': lr[3] * decay})
                params.append({'params': [self.ai6], 'lr': lr[3] * decay**2})
                params.append({'params': [self.ai8], 'lr': lr[3] * decay**3})
                params.append({'params': [self.ai10], 'lr': lr[3] * decay**4})
            elif self.ai_degree == 6:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
                self.ai10.requires_grad_(True)
                self.ai12.requires_grad_(True)
                params.append({'params': [self.ai2], 'lr': lr[3]})
                params.append({'params': [self.ai4], 'lr': lr[3] * decay})
                params.append({'params': [self.ai6], 'lr': lr[3] * decay**2})
                params.append({'params': [self.ai8], 'lr': lr[3] * decay**3})
                params.append({'params': [self.ai10], 'lr': lr[3] * decay**4})
                params.append({'params': [self.ai12], 'lr': lr[3] * decay**5})
            else:
                for i in range(1, self.ai_degree + 1):
                    exec(f'self.ai{2*i}.requires_grad_(True)')
                    exec(f'params.append({{\'params\': [self.ai{2*i}], \'lr\': lr[3] * decay**{i-1}}})')

        return params


    @torch.no_grad()
    def perturb(self, ratio=0.001, thickness_precision=0.0005, diameter_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors. This function should only be called in the final image simulation stage. 
        
        Args:
            ratio (float, optional): perturbation ratio. Defaults to 0.001.
            thickness_precision (float, optional): thickness precision. Defaults to 0.0005.
            diameter_precision (float, optional): diameter precision. Defaults to 0.001.
        """
        self.r += np.random.randn() * diameter_precision
        if self.c != 0:
            self.c *= 1 + np.random.randn() * ratio
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision
        if self.k != 0:
            self.k *= 1 + np.random.randn() * ratio
        for i in range(1, self.ai_degree+1):
            exec(f'self.ai{2*i} *= 1 + np.random.randn() * ratio')


    def surf_dict(self):
        """ Return a dict of surface.
        """
        surf_dict = {
            'type': 'Aspheric',
            'r': self.r,
            'c': self.c.item(),
            'roc': 1 / self.c.item(),
            'd': self.d.item(),
            'k': self.k.item(),
            'ai': [],
            # 'mat1': self.mat1.name,
            'mat2': self.mat2.name,
            }
        for i in range(1, self.ai_degree+1):
            exec(f'surf_dict[\'ai{2*i}\'] = self.ai{2*i}.item()')
            surf_dict['ai'].append(eval(f'self.ai{2*i}.item()'))

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        assert self.c.item() != 0, 'Aperture surface is re-implemented in Aperture class.'
        assert self.ai is not None or self.k != 0, 'Spheric surface is re-implemented in Spheric class.'
        if self.mat2.name == 'air':
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH
    CURV {self.c.item()} 
    DISZ {self.d.item()}
    DIAM {self.r * 2}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE EVENASPH 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS {self.mat2.name.upper()} 0 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r * 2}
    PARM 1 {self.ai2.item()}
    PARM 2 {self.ai4.item()}
    PARM 3 {self.ai6.item()}
    PARM 4 {self.ai8.item()}
    PARM 5 {self.ai10.item()}
    PARM 6 {self.ai12.item()}
"""
        return zmx_str


class Cubic(Surface):
    """ Cubic surface: z(x,y) = b3 * (x**3 + y**3)

        Actually Cubic phase is a group of surfaces with changing height.

        Can also be written as: f(x, y, z) = 0
    """
    def __init__(self, r, d, ai, mat2, is_square=False, device=DEVICE):
        Surface.__init__(self, r, d, mat2, is_square=is_square, device=device) 
        self.ai = torch.Tensor(ai)

        if len(ai) == 1:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b_degree = 1
        elif len(ai) == 2:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b5 = torch.Tensor([ai[1]]).to(device)
            self.b_degree = 2
        elif len(ai) == 3:
            self.b3 = torch.Tensor([ai[0]]).to(device)
            self.b5 = torch.Tensor([ai[1]]).to(device)
            self.b7 = torch.Tensor([ai[2]]).to(device)
            self.b_degree = 3
        else:
            raise Exception('Unsupported cubic degree!!')

        self.rotate_angle = 0.0

    def g(self, x, y):
        """ Compute surface height z(x, y).
        """
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5) + self.b7 * (x**7 + y**7)
        else:
            raise Exception('Unsupported cubic degre!')
        
        if len(z.size()) == 0:
            z = torch.Tensor([z]).to(self.device)

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)
        
        return z

    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 1:
            sx = 3 * self.b3 * x**2
            sy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            sx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise Exception('Unsupported cubic degree!')

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        return sx, sy

    def get_optimizer_params(self, lr):
        """ Return parameters for optimizer.
        """
        params = []
        
        self.d.requires_grad_(True)
        params.append({'params': [self.d], 'lr': lr})
        
        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({'params': [self.b3], 'lr': lr})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({'params': [self.b3], 'lr': lr})
            params.append({'params': [self.b5], 'lr': lr * 0.1})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({'params': [self.b3], 'lr': lr})
            params.append({'params': [self.b5], 'lr': lr * 0.1})
            params.append({'params': [self.b7], 'lr': lr * 0.01})
        else:
            raise Exception('Unsupported cubic degree!')
        
        return params

    def perturb(self, curvature_precision=0.001, thickness_precision=0.0005, diameter_precision=0.01, angle=0.01):
        """ Perturb the surface
        """
        self.r += np.random.randn() * diameter_precision
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision

        if self.b_degree == 1:
            self.b3 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 2:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 3:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
            self.b7 *= 1 + np.random.randn() * curvature_precision

        self.rotate_angle = np.random.randn() * angle

    def surf_dict(self):
        """ Return surface parameters.
        """
        return {
            'type': 'cubic',
            'b3': self.b3,
            'b5': self.b5,
            'b7': self.b7,
            'r': self.r,
            'd': self.d,
            'rotate_angle': self.rotate_angle
        }


class DOE_GEO(Surface):
    """ Kinoform and binary diffractive surfaces for ray tracing.

        https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
    """
    def __init__(self, l, d, thickness=0.5, glass='test', param_model='binary2', device=DEVICE):
        Surface.__init__(self, l / np.sqrt(2), d, mat2='air', is_square=True, device=device)

        # DOE geometry
        self.w, self.h = l, l
        self.r = l / float(np.sqrt(2))
        self.l = l
        self.thickness = thickness
        self.glass = glass

        # Use ray tracing to simulate diffraction, the same as Zemax
        self.diffraction = False
        self.diffraction_order = 1
        print('DOE_GEO initialization: diffraction is not activated.')
        
        self.to(device)
        self.init_param_model(param_model)


    def init_param_model(self, param_model='binary2'):
        self.param_model = param_model
        if self.param_model == 'fresnel':
            # Focal length at 550nm
            self.f0 = torch.tensor([100.0])
        
        elif self.param_model == 'binary2':
            # Zemax binary2 surface type
            self.order2 = torch.tensor([0.0])
            self.order4 = torch.tensor([0.0])
            self.order6 = torch.tensor([0.0])
            self.order8 = torch.tensor([0.0])
        
        elif self.param_model == 'poly1d':
            rand_value = np.random.rand(6) * 0.001
            self.order2 = torch.tensor(rand_value[0])
            self.order3 = torch.tensor(rand_value[1])
            self.order4 = torch.tensor(rand_value[2])
            self.order5 = torch.tensor(rand_value[3])
            self.order6 = torch.tensor(rand_value[4])
            self.order7 = torch.tensor(rand_value[5])
        
        elif self.param_model == 'grating':
            # A grating surface
            self.theta = torch.tensor([0.0])    # angle from x-axis to grating vector
            self.alpha = torch.tensor([0.0])    # slope of the grating
        
        else: 
            raise Exception('Unsupported parameter model!')

        self.to(self.device)


    def activate_diffraction(self, diffraction_order=1):
        self.diffraction = True
        self.diffraction_order = diffraction_order
        print('Diffraction of DOE in ray tracing is enabled.')

    # ==============================
    # Computation (ray tracing) 
    # ==============================
    def ray_reaction(self, ray, n1=1.0, n2=1.0):
        """ Ray reaction on DOE surface. Imagine the DOE as a wrapped positive convex lens for debugging. 

            1, The phase φ in radians adds to the optical path length of the ray
            2, The gradient of the phase profile (phase slope) change the direction of rays.

            https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        """
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0

        # Intersection
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        # valid = (new_o[...,0].abs() <= self.h/2) & (new_o[...,1].abs() <= self.w/2) & (ray.ra > 0) # square
        valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) <= self.r) & (ray.ra > 0)    # circular
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            # OPL change
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        if self.diffraction:
            # Diffraction 1: DOE phase modulation
            if ray.coherent:
                phi = self.phi(ray.o[...,0], ray.o[...,1])
                new_opl = ray.opl + phi * (ray.wvln * 1e-3) / (2 * np.pi)
                new_opl[~valid] = ray.opl[~valid]
                ray.opl = new_opl

            # Diffraction 2: bend rays
            # Perpendicular incident rays are diffracted following (1) grating equation and (2) local grating approximation
            dphidx, dphidy = self.dphi_dxy(ray.o[...,0], ray.o[...,1])

            if forward:
                new_d_x = ray.d[..., 0] + (ray.wvln * 1e-3) / (2 * np.pi) * dphidx * self.diffraction_order
                new_d_y = ray.d[..., 1] + (ray.wvln * 1e-3) / (2 * np.pi) * dphidy * self.diffraction_order
            else:
                new_d_x = ray.d[..., 0] - (ray.wvln * 1e-3) / (2 * np.pi) * dphidx * self.diffraction_order
                new_d_y = ray.d[..., 1] - (ray.wvln * 1e-3) / (2 * np.pi) * dphidy * self.diffraction_order

            new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
            new_d = nnF.normalize(new_d, p=2, dim=-1)
            
            new_d[~valid] = ray.d[~valid]
            ray.d = new_d

        return ray
    
    def phi(self, x, y):
        """ Reference phase map at design wavelength (independent to wavelength). We have the same definition of phase (phi) as Zemax.
        """
        x_norm = x / self.r
        y_norm = y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)
        
        if self.param_model == 'fresnel':
            phi = - 2 * np.pi * torch.fmod((x**2 + y**2) / (2 * 0.55e-3 * self.f0), 1) # unit [mm]
        
        elif self.param_model == 'binary2':
            phi = self.order2 * r**2 + self.order4 * r**4 + self.order6 * r**6 + self.order8 * r**8
        
        elif self.param_model == 'poly1d':
            phi_even = self.order2 * r**2 + self.order4 * r**4 + self.order6 * r**6
            phi_odd =  self.order3 * (x_norm**3 + y_norm**3) + self.order5 * (x_norm**5 + y_norm**5) + self.order7 * (x_norm**7 + y_norm**7)
            phi = phi_even + phi_odd

        elif self.param_model == 'grating':
            phi = self.alpha * (x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta))
        
        else:
            raise NotImplementedError
        
        phi = torch.remainder(phi, 2 * np.pi)
        return phi
    
    def dphi_dxy(self, x, y):
        """ Calculate phase derivatives (dphi/dx, dphi/dy) for given points.
        """
        x_norm = x / self.r
        y_norm = y / self.r
        r = torch.sqrt(x_norm**2 + y_norm**2 + EPSILON)
        
        if self.param_model == 'fresnel':
            dphidx = - 2 * np.pi * x / (0.55e-3 * self.f0) # unit [mm]
            dphidy = - 2 * np.pi * y / (0.55e-3 * self.f0)
        
        elif self.param_model == 'binary2':
            dphidr = 2 * self.order2 * r + 4 * self.order4 * r**3 + 6 * self.order6 * r**5 + 8 * self.order8 * r**7
            dphidx = dphidr * x_norm / r / self.r
            dphidy = dphidr * y_norm / r / self.r
        
        elif self.param_model == 'poly1d':
            dphi_even_dr = 2 * self.order2 * r + 4 * self.order4 * r**3 + 6 * self.order6 * r**5
            dphi_even_dz = dphi_even_dr * x_norm / r / self.r
            dphi_even_dy = dphi_even_dr * y_norm / r / self.r
            
            dphi_odd_dx =  (3 * self.order3 * x_norm**2 + 5 * self.order5 * x_norm**4 + 7 * self.order7 * x_norm**6) / self.r
            dphi_odd_dy =  (3 * self.order3 * y_norm**2 + 5 * self.order5 * y_norm**4 + 7 * self.order7 * y_norm**6) / self.r
            
            dphidx = dphi_even_dz + dphi_odd_dx
            dphidy = dphi_even_dy + dphi_odd_dy

        elif self.param_model == 'grating':
            dphidx = self.alpha * torch.sin(self.theta) / self.r
            dphidy = self.alpha * torch.cos(self.theta) / self.r
        
        else:
            raise NotImplementedError

        return dphidx, dphidy
    
    def g(self, x, y):
        raise Exception('self.g() function is meaningless for phase DOE, use self.phi() function.')
    
    def dgd(self, x, y):
        raise Exception('self.dgd() function is meaningless for phase DOE, use self.dphidxy() function.')

    def surface(self, x, y, max_offset=0.2):
        """ When drawing the lens setup, this function is called to compute the surface height.

            Here we use a fake height ONLY for drawing.
        """    
        roc = self.l
        r = torch.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - torch.sqrt(1 - r**2 / roc**2))
        sag = max_offset - torch.fmod(sag, max_offset)
        return sag
    
    def draw_phase_map(self, save_name='./DOE_phase_map.png'):
        """ Draw height map. Range from [0, max_height].
        """
        x, y = torch.meshgrid(torch.linspace(-self.l/2, self.l/2, 2000), torch.linspace(self.l/2, -self.l/2, 2000), indexing='xy')
        x, y = x.to(self.device), y.to(self.device)
        pmap = self.phi(x, y)
        # pmap_q = self.pmap_quantize()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * np.pi)
        ax[0].set_title(f'Phase map 0.55um', fontsize=10)
        ax[0].grid(False)
        fig.colorbar(ax[0].get_images()[0])
        
        # ax[1].imshow(pmap_q.cpu().numpy(), vmin=0, vmax=2 * np.pi)
        # ax[1].set_title(f'Quantized phase map ({self.wvln0}um)', fontsize=10)
        # ax[1].grid(False)
        # fig.colorbar(ax[1].get_images()[0])

        fig.savefig(save_name, dpi=600, bbox_inches='tight')
        plt.close(fig)

    # ==============================
    # Optimization and other functions
    # ==============================
    def activate_grad(self, activate=True):
        """ Activate gradient for all parameters.
        """
        if self.param_model == 'binary2':
            self.order2.requires_grad = activate
            self.order4.requires_grad = activate
            self.order6.requires_grad = activate
        elif self.param_model == 'poly1d':
            self.order2.requires_grad = activate
            self.order3.requires_grad = activate
            self.order4.requires_grad = activate
            self.order5.requires_grad = activate
            self.order6.requires_grad = activate
            self.order7.requires_grad = activate
        elif self.param_model == 'grating':
            self.theta.requires_grad = activate
            self.alpha.requires_grad = activate
        else:
            raise NotImplementedError
        
    def get_optimizer_params(self, lr=None):
        """ Generate optimizer parameters.
        """
        self.activate_grad()
        params = []
        if self.param_model == 'binary2':
            lr = 0.001 if lr is None else lr
            params.append({'params': [self.order2], 'lr': lr})
            params.append({'params': [self.order4], 'lr': lr})
            params.append({'params': [self.order6], 'lr': lr})
            params.append({'params': [self.order8], 'lr': lr})

        elif self.param_model == 'poly1d':
            lr = 0.001 if lr is None else lr
            params.append({'params': [self.order2], 'lr': lr})
            params.append({'params': [self.order3], 'lr': lr})
            params.append({'params': [self.order4], 'lr': lr})
            params.append({'params': [self.order5], 'lr': lr})
            params.append({'params': [self.order6], 'lr': lr})
            params.append({'params': [self.order7], 'lr': lr})

        elif self.param_model == 'grating':
            lr = 0.1 if lr is None else lr
            params.append({'params': [self.theta], 'lr': lr})
            params.append({'params': [self.alpha], 'lr': lr})

        else:
            raise NotImplementedError
        
        return params

    def get_optimizer(self, lr=None):
        """ Generate optimizer.

        Args:
            lr (float, optional): Learning rate. Defaults to 1e-3.
            iterations (float, optional): Iterations. Defaults to 1e4.
        """
        assert self.diffraction, 'Diffraction is not activated yet.'
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)
        return optimizer

    def save_ckpt(self, save_path='./doe.pth'):
        """ Save DOE height map.
        """
        if self.param_model == 'binary2':
            torch.save({
                'param_model': self.param_model,
                'order2': self.order2.clone().detach().cpu(),
                'order4': self.order4.clone().detach().cpu(),
                'order6': self.order6.clone().detach().cpu(),
                'order8': self.order8.clone().detach().cpu(),
            }, save_path
            )
        elif self.param_model == 'poly1d':
            torch.save({
                'param_model': self.param_model,
                'order2': self.order2.clone().detach().cpu(),
                'order3': self.order3.clone().detach().cpu(),
                'order4': self.order4.clone().detach().cpu(),
                'order5': self.order5.clone().detach().cpu(),
                'order6': self.order6.clone().detach().cpu(),
                'order7': self.order7.clone().detach().cpu(),
            }, save_path
            )
        elif self.param_model == 'grating':
            torch.save({
                'param_model': self.param_model,
                'theta': self.theta.clone().detach().cpu(),
                'alpha': self.alpha.clone().detach().cpu(),
            }, save_path
            )
        else:
            raise Exception('Unknown parameterization.')

    def load_ckpt(self, load_path='./doe.pth'):
        """ Load DOE height map.
        """
        self.diffraction = True
        ckpt = torch.load(load_path)
        self.param_model = ckpt['param_model']
        if self.param_model == 'binary2' or self.param_model == 'poly_even':
            self.param_model = 'binary2'
            self.order2 = ckpt['order2'].to(self.device)
            self.order4 = ckpt['order4'].to(self.device)
            self.order6 = ckpt['order6'].to(self.device)
            self.order8 = ckpt['order8'].to(self.device)
        elif self.param_model == 'poly1d':
            self.order2 = ckpt['order2'].to(self.device)
            self.order3 = ckpt['order3'].to(self.device)
            self.order4 = ckpt['order4'].to(self.device)
            self.order5 = ckpt['order5'].to(self.device)
            self.order6 = ckpt['order6'].to(self.device)
            self.order7 = ckpt['order7'].to(self.device)
        elif self.param_model == 'grating':
            self.theta = ckpt['theta'].to(self.device)
            self.alpha = ckpt['alpha'].to(self.device)
        else:
            raise Exception('Unknown parameterization.')
        
    def surf_dict(self):
        """ Return surface parameters.
        """
        if self.param_model == 'fresnel':
            surf_dict = {
                'type': self.__class__.__name__,
                'l': self.l,
                'glass': self.glass,
                'param_model': self.param_model,
                'f0': self.f0.item(),
                'd': self.d.item(),
                # 'mat1': self.mat1.name,
                'mat2': self.mat2.name,
            }

        elif self.param_model == 'binary2':
            surf_dict = {
                'type': self.__class__.__name__,
                'l': self.l,
                'glass': self.glass,
                'param_model': self.param_model,
                'order2': self.order2.item(),
                'order4': self.order4.item(),
                'order6': self.order6.item(),
                'order8': self.order8.item(),
                'd': self.d.item(),
                # 'mat1': self.mat1.name,
                'mat2': self.mat2.name,
            }

        elif self.param_model == 'poly1d':
            surf_dict = {
                'type': self.__class__.__name__,
                'l': self.l,
                'glass': self.glass,
                'param_model': self.param_model,
                'order2': self.order2.item(),
                'order3': self.order3.item(),
                'order4': self.order4.item(),
                'order5': self.order5.item(),
                'order6': self.order6.item(),
                'order7': self.order7.item(),
                'd': self.d.item(),
                # 'mat1': self.mat1.name,
                'mat2': self.mat2.name,
            }

        elif self.param_model == 'grating':
            surf_dict = {
                'type': self.__class__.__name__,
                'l': self.l,
                'glass': self.glass,
                'param_model': self.param_model,
                'theta': self.theta.item(),
                'alpha': self.alpha.item(),
                'd': self.d.item(),
                # 'mat1': self.mat1.name,
                'mat2': self.mat2.name,
            }

        return surf_dict

class Freeform(Surface):
    def __init__(self, l, d, mat2, device=DEVICE):
        """ Freeform surface.

            Can be either zernike, or xy polynomial (easy to fabricate), or B-spline.
        """
        Surface.__init__(self, l / np.sqrt(2), d, mat2=mat2, is_square=True, device=device)
        self.l = l
        self.d = d
        self.mat2 = mat2
        self.to(device)

class Plane(Surface):
    def __init__(self, l, d, mat2, is_square=True, device=DEVICE):
        """ Plane surface, typically rectangle. Working as IR filter, lens cover glass or DOE base.
        """
        Surface.__init__(self, l / np.sqrt(2), d, mat2=mat2, is_square=is_square, device=device)
        
    def intersect(self, ray, n=1.0):
        """ Solve ray-surface intersection and update ray data.
        """
        # Solve intersection
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        if self.is_square:
            valid = (torch.abs(new_o[...,0]) < self.w/2) & (torch.abs(new_o[...,1]) < self.h/2) & (ray.ra > 0)
        else:
            valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) < self.r) & (ray.ra > 0)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + n * t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def normal(self, ray):
        """ Calculate surface normal vector at intersection points.
        """
        n = torch.zeros_like(ray.d)
        n[...,2] = -1
        return n

    def g(self, x, y):
        return torch.zeros_like(x)
    
    def dgd(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)
    
    def surf_dict(self):
        surf_dict = {
            'type': "Plane",
            'l': self.l,
            'd': self.d.item(),
            'is_square': True
        }

        return surf_dict
    
class Spheric(Surface):
    """ Spheric surface.
    """
    def __init__(self, c, r, d, mat2, device=DEVICE):
        super(Spheric, self).__init__(r, d, mat2, is_square=False, device=device)
        self.c = torch.tensor([c])

        self.c_perturb = 0.0
        self.d_perturb = 0.0
        self.to(device)

    def g(self, x, y):
        """ Compute surfaces sag z = r**2 * c / (1 - sqrt(1 - r**2 * c**2))
        """
        c = self.c + self.c_perturb

        r2 = x**2 + y**2
        sag = c * r2 / (1 + torch.sqrt(1 - r2 * c**2))
        return sag

    def dgd(self, x, y):
        """ Compute surface sag derivatives to x and y: dz / dx, dz / dy.
        """
        c = self.c + self.c_perturb

        r2 = x**2 + y**2
        sf = torch.sqrt(1 - r2 * c**2 + EPSILON)
        dgdr2 =  c / (2 * sf)
        return dgdr2*2*x, dgdr2*2*y

    def valid(self, x, y):
        """ Invalid when shape is non-defined.
        """
        c = self.c + self.c_perturb

        valid = ((x**2 + y**2) < 1 / c**2)
        return valid
    
    def max_height(self):
        """ Maximum valid height.
        """
        c = self.c + self.c_perturb

        max_height = torch.sqrt(1 / c**2).item() - 0.01
        return max_height
    
    def perturb(self, d_precision=0.001, c_precision=0.001):
        """ Randomly perturb surface parameters to simulate manufacturing errors.
        """
        self.c_perturb = self.c.item() * float(np.random.randn() * c_precision)
        self.d_perturb = float(np.random.randn() * d_precision)

    def no_perturb(self):
        """ Reset perturbation.
        """
        self.c_perturb = 0.0
        self.d_perturb = 0.0
    
    def get_optimizer_params(self, lr=[0.001, 0.001]):
        """ Activate gradient computation for c and d and return optimizer parameters.
        """
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)
        
        params = []
        params.append({'params': [self.c], 'lr': lr[0]})
        params.append({'params': [self.d], 'lr': lr[1]})
        return params

    def surf_dict(self):
        """ Return surface parameters.
        """
        roc = 1 / self.c.item() if self.c.item() != 0 else 0.0
        surf_dict = {
                'type': 'Spheric',
                'r': self.r,
                'c': self.c.item(),
                'roc': roc,
                'd': self.d.item(),
                # 'mat1': self.mat1.name,
                'mat2': self.mat2.name,
                }

        return surf_dict
    
    def zmx_str(self, surf_idx, d_next):
        """ Return Zemax surface string.
        """
        if self.mat2.name == 'air':
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r*2}
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS {self.mat2.name.upper()} 0 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r*2}
"""

        return zmx_str


class ThinLens(Surface):
    def __init__(self, f, r, d, mat2='air', is_square=False, device=DEVICE):
        """ Thin lens surface. 
        """
        Surface.__init__(self, r, d, mat2=mat2, is_square=is_square, device=device)
        self.f = torch.tensor([f])

    def intersect(self, ray, n=1.0):
        """ Solve ray-surface intersection and update rays.
        """
        # Solve intersection
        t = (self.d - ray.o[...,2]) / ray.d[...,2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        valid = (torch.sqrt(new_o[...,0]**2 + new_o[...,1]**2) < self.r) & (ray.ra > 0)

        # Update rays
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        
        new_o[~valid] = ray.o[~valid]
        ray.o = new_o
        ray.ra = ray.ra * valid

        if ray.coherent:
            new_opl = ray.opl + t
            new_opl[~valid] = ray.opl[~valid]
            ray.opl = new_opl

        return ray
    
    def refract(self, ray, n=1.0):
        """ For a thin lens, all rays will converge to z = f plane. Therefore we trace the chief-ray (parallel-shift to surface center) to find the final convergence point for each ray. 
        
            For coherent ray tracing, we can think it as a Fresnel lens with infinite refractive index.
            (1) Lens maker's equation
            (2) Spherical lens function
        """
        forward = (ray.d * ray.ra.unsqueeze(-1))[...,2].sum() > 0

        # Calculate convergence point
        if forward:
            t0 = self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = torch.full_like(xy_final[..., 0].unsqueeze(-1), self.d.item() + self.f.item())
            o_final = torch.cat([xy_final, z_final], dim=-1)
        else:
            t0 = - self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = torch.full_like(xy_final[..., 0].unsqueeze(-1), self.d.item() - self.f.item())
            o_final = torch.cat([xy_final, z_final], dim=-1)

        # New ray direction
        new_d = o_final - ray.o
        new_d = nnF.normalize(new_d, p=2, dim=-1)
        ray.d = new_d

        # OPL change
        if ray.coherent:
            if forward:
                ray.opl = ray.opl - (ray.o[..., 0]**2 + ray.o[..., 1]**2) / self.f / 2 / ray.d[..., 2]
            else:
                ray.opl = ray.opl + (ray.o[..., 0]**2 + ray.o[..., 1]**2) / self.f / 2 / ray.d[..., 2]
        
        return ray
    
    def g(self, x, y):
        return torch.zeros_like(x)
    
    def dgd(self, x, y):
        return torch.zeros_like(x), torch.zeros_like(x)
    
    # def surface(self, x, y):
    #     if torch.is_tensor(x):
    #         return torch.zeros_like(x)
    #     else:
    #         return 0
    

# class Freeform(Surface):
#     """ Freeform surface represented by several pedal surfaces.

#         In Zemax:
#         https://support.zemax.com/hc/zh-cn/articles/1500005576322-%E5%9C%A8OpticStudio%E4%B8%AD%E4%BD%BF%E7%94%A8%E8%87%AA%E7%94%B1%E6%9B%B2%E9%9D%A2%E9%80%B2%E8%A1%8C%E8%A8%AD%E8%A8%88

#         Pedal surface: S(r) = b - sqrt((b^2 - 2r^2 + sqrt(b^4 + 4(a^2-b^2)r^2)) / 2)
#         Freeform surface: z(r) = \sum alpha1 * S1 + alpha2 * S1^2 + alpha3 * S1^3

#         a, b: [2 * 1]
#         alpha1, alpha2, alpha3: [2 * 1]
#     """
#     def __init__(self, r, d, pedal_degree=2, a=[0, 0], b=[0, 0], alpha1=[0, 0], alpha2=[0, 0], alpha3=[0, 0], is_square=False, device=DEVICE):
#         Surface.__init__(self, r, d, is_square, device)

#         self.pedal_degree = pedal_degree
#         self.a = torch.Tensor(np.array(a)) if a is not None else torch.zeros((pedal_degree, 1), device=device)
#         self.b = torch.Tensor(np.array(b)) if b is not None else torch.zeros((pedal_degree, 1), device=device)
#         self.alpha1 = torch.Tensor(np.array(alpha1)) if alpha1 is not None else torch.zeros((pedal_degree, 1), device=device)
#         self.alpha2 = torch.Tensor(np.array(alpha2)) if alpha2 is not None else torch.zeros((pedal_degree, 1), device=device)
#         self.alpha3 = torch.Tensor(np.array(alpha3)) if alpha3 is not None else torch.zeros((pedal_degree, 1), device=device)
    

#     def init_alpha(self):
#         """ Init alpha.

#             Reference value: alpha1 = -0.61, alpha2 = 0.0036, alpha3 = -0.0008
#         """
#         self.alpha1 = -torch.rand((self.pedal_degree, 1), device=self.device) * 0.1    # range [-0.1, 0]
#         self.alpha2 = (torch.rand((self.pedal_degree, 1), device=self.device) - 0.5) * 0.02    # range in [-0.001, 0.001]
#         self.alpha3 = (torch.rand((self.pedal_degree, 1), device=self.device) - 0.5) * 0.002    # range in [-0.0001, 0.0001]
#         return


#     def init_ab(self):
#         """ Init a and b.

#             Reference value: a = 2, b = 1 from "Miniature camera lens design with a freeform surface"
#         """
#         self.a = torch.rand((self.pedal_degree, 1) * 1, device=self.device)
#         self.b = torch.rand((self.pedal_degree, 1) * 1, device=self.device)
#         return


    
#     # ==============================
#     # Surface and derivatives.
#     # ==============================
#     def g(self, x, y, valid=None):
#         """ Compute z(r), called by outerfunctions.
#         """
#         if valid is None:
#             valid = self.valid(x, y)
#         else:
#             valid = valid * self.valid(x, y)

#         x, y = x*valid, y*valid
#         return self._z(x, y)


#     def _z(self, x, y):
#         """ Compute z(r).

#             Original _g function.
#         """
#         r2 = x**2 + y**2
#         S1, S2 = self._pedal(r2)
#         z = self.alpha1[0] * S1 + self.alpha2[0] * S1**2 + self.alpha3[0] * S1**3 + self.alpha1[1] * S2 + self.alpha2[1] * S2**2 + self.alpha3[1] * S2**3

#         return z


#     def _pedal(self, r2):
#         """ Compute sag of two pedal surfaces.

#             TODO: support higher pedal degree.
#         """
#         S1 = self.b[0] - torch.sqrt((self.b[0]**2 - 2*r2 + torch.sqrt(self.b[0]**4 + 4*(self.a[0]**2-self.b[0]**2)*r2))/2)
#         S2 = self.b[1] - torch.sqrt((self.b[1]**2 - 2*r2 + torch.sqrt(self.b[1]**4 + 4*(self.a[1]**2-self.b[1]**2)*r2))/2)
#         return S1, S2


#     def _dzd(self, x, y):
#         """ Compute dz / dr2
#         """

#         return 


#     def valid(self, x, y):
#         """ Check validity.
#         """
#         valid = (x**2 + y**2) < self.r**2
#         return valid

# class BSpline(Surface):
#     """
#     Implemented according to Wikipedia.
#     """
#     def __init__(self, r, d, size, px=3, py=3, tx=None, ty=None, c=None, is_square=False, device=DEVICE): # input c is 1D
#         Surface.__init__(self, r, d, is_square, device)
#         self.px = px
#         self.py = py
#         self.size = np.asarray(size)

#         # knots
#         if tx is None:
#             self.tx = None
#         else:
#             if len(tx) != size[0] + 2*(self.px + 1):
#                 raise Exception('len(tx) is not correct!')
#             self.tx = torch.Tensor(np.asarray(tx)).to(self.device)
#         if ty is None:
#             self.ty = None
#         else:
#             if len(ty) != size[1] + 2*(self.py + 1):
#                 raise Exception('len(ty) is not correct!')
#             self.ty = torch.Tensor(np.asarray(ty)).to(self.device)

#         # c is the only differentiable parameter
#         c_shape = size + np.array([self.px, self.py]) + 1
#         if c is None:
#             self.c = None
#         else:
#             c = np.asarray(c)
#             if c.size != np.prod(c_shape):
#                 raise Exception('len(c) is not correct!')
#             self.c = torch.Tensor(c.reshape(*c_shape)).to(self.device)
        
#         if (self.tx is None) or (self.ty is None) or (self.c is None):
#             self.tx = self._generate_knots(self.r, size[0], p=px, device=device)
#             self.ty = self._generate_knots(self.r, size[1], p=py, device=device)
#             self.c = torch.zeros(*c_shape, device=device)
#         else:
#             self.to(self.device)

#     @staticmethod
#     def _generate_knots(R, n, p=3, device=torch.device('cpu')):
#         t = np.linspace(-R, R, n)
#         step = t[1] - t[0]
#         T = t[0] - 0.9 * step
#         np.pad(t, p+1, 'constant', constant_values=step)
#         t = np.concatenate((np.ones(p+1)*T, t, -np.ones(p+1)*T), axis=0)
#         return torch.Tensor(t).to(device)

#     def fit(self, x, y, z, eps=1e-3):
#         x, y, z = (v.flatten() for v in [x, y, z])

#         # knot positions within [-r, r]^2
#         X = np.linspace(-self.r, self.r, self.size[0])
#         Y = np.linspace(-self.r, self.r, self.size[1])
#         bs = LSQBivariateSpline(x, y, z, X, Y, kx=self.px, ky=self.py, eps=eps)
#         # print('RMS residual error is {} um'.format(np.sqrt(bs.fp/len(z))*1e3))
#         tx, ty = bs.get_knots()
#         c = bs.get_coeffs().reshape(len(tx)-self.px-1, len(ty)-self.py-1)

#         # convert to torch.Tensor
#         self.tx, self.ty, self.c = (torch.Tensor(v).to(self.device) for v in [tx, ty, c])

#     # === Common methods
#     def g(self, x, y):
#         return self._deBoor2(x, y)

#     def dgd(self, x, y):
#         return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1)

#     def h(self, z):
#         return -z

#     def dhd(self, z):
#         return -torch.ones_like(z)

#     def surface(self, x, y):
#         return self._deBoor2(x, y)

#     def surface_derivatives(self, x, y):
#         return self._deBoor2(x, y, dx=1), self._deBoor2(x, y, dy=1), -torch.ones_like(x)

#     def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z):
#         #pylint: disable=unused-argument
#         x = ox + t * dx
#         y = oy + t * dy
#         s, sx, sy = self._deBoor2(x, y, dx=-1, dy=-1)
#         return s - z, sx*dx + sy*dy - dz

#     def reverse(self):
#         self.c = -self.c

#     # === Private methods
#     def _deBoor(self, x, t, c, p=3, is2Dfinal=False, dx=0):
#         """
#         Arguments
#         ---------
#         x: Position.
#         t: Array of knot positions, needs to be padded as described above.
#         c: Array of control points.
#         p: Degree of B-spline.
#         dx:
#         - 0: surface only
#         - 1: surface 1st derivative only
#         - -1: surface and its 1st derivative
#         """
#         k = torch.sum((x[None,...] > t[...,None]).int(), axis=0) - (p+1)
        
#         if is2Dfinal:
#             inds = np.indices(k.shape)[0]
#             def _c(jk): return c[jk, inds]
#         else:
#             def _c(jk): return c[jk, ...]

#         need_newdim = (len(c.shape) > 1) & (not is2Dfinal)

#         def f(a, b, alpha):
#             if need_newdim:
#                 alpha = alpha[...,None]
#             return (1.0 - alpha) * a + alpha * b
        
#         # surface only
#         if dx == 0:
#             d = [_c(j+k) for j in range(0, p+1)]

#             for r in range(-p, 0):
#                 for j in range(p, p+r, -1):
#                     left = j+k
#                     t_left  = t[left]
#                     t_right = t[left-r]
#                     alpha = (x - t_left) / (t_right - t_left)
#                     d[j] = f(d[j-1], d[j], alpha)
#             return d[p]

#         # surface 1st derivative only
#         if dx == 1:
#             q = []
#             for j in range(1, p+1):
#                 jk = j+k
#                 tmp = t[jk+p] - t[jk]
#                 if need_newdim:
#                     tmp = tmp[..., None]
#                 q.append(p * (_c(jk) - _c(jk-1)) / tmp)

#             for r in range(-p, -1):
#                 for j in range(p-1, p+r, -1):
#                     left = j+k
#                     t_right = t[left-r]
#                     t_left_ = t[left+1]
#                     alpha = (x - t_left_) / (t_right - t_left_)
#                     q[j] = f(q[j-1], q[j], alpha)
#             return q[p-1]
            
#         # surface and its derivative (all)
#         if dx < 0:
#             d, q = [], []
#             for j in range(0, p+1):
#                 jk = j+k
#                 c_jk = _c(jk)
#                 d.append(c_jk)
#                 if j > 0:
#                     tmp = t[jk+p] - t[jk]
#                     if need_newdim:
#                         tmp = tmp[..., None]
#                     q.append(p * (c_jk - _c(jk-1)) / tmp)

#             for r in range(-p, 0):
#                 for j in range(p, p+r, -1):
#                     left = j+k
#                     t_left  = t[left]
#                     t_right = t[left-r]
#                     alpha = (x - t_left) / (t_right - t_left)
#                     d[j] = f(d[j-1], d[j], alpha)

#                     if (r < -1) & (j < p):
#                         t_left_ = t[left+1]
#                         alpha = (x - t_left_) / (t_right - t_left_)
#                         q[j] = f(q[j-1], q[j], alpha)
#             return d[p], q[p-1]

#     def _deBoor2(self, x, y, dx=0, dy=0):
#         """
#         Arguments
#         ---------
#         x,  y : Position.
#         dx, dy: 
#         """
#         if not torch.is_tensor(x):
#             x = torch.Tensor(np.asarray(x)).to(self.device)
#         if not torch.is_tensor(y):
#             y = torch.Tensor(np.asarray(y)).to(self.device)
#         dim = x.shape

#         x = x.flatten()
#         y = y.flatten()

#         # handle boundary issue
#         x = torch.clamp(x, min=-self.r, max=self.r)
#         y = torch.clamp(y, min=-self.r, max=self.r)

#         if (dx == 0) & (dy == 0):     # spline
#             s_tmp = self._deBoor(x, self.tx, self.c, self.px)
#             s = self._deBoor(y, self.ty, s_tmp.T, self.py, True)
#             return s.reshape(dim)
#         elif (dx == 1) & (dy == 0):  # x-derivative
#             s_tmp = self._deBoor(y, self.ty, self.c.T, self.py)
#             s_x = self._deBoor(x, self.tx, s_tmp.T, self.px, True, dx)
#             return s_x.reshape(dim)
#         elif (dy == 1) & (dx == 0):  # y-derivative
#             s_tmp = self._deBoor(x, self.tx, self.c, self.px)
#             s_y = self._deBoor(y, self.ty, s_tmp.T, self.py, True, dy)
#             return s_y.reshape(dim)
#         else:                       # return all
#             s_tmpx = self._deBoor(x, self.tx, self.c, self.px)
#             s_tmpy = self._deBoor(y, self.ty, self.c.T, self.py)
#             s, s_x = self._deBoor(x, self.tx, s_tmpy.T, self.px, True, -abs(dx))
#             s_y = self._deBoor(y, self.ty, s_tmpx.T, self.py, True, abs(dy))
#             return s.reshape(dim), s_x.reshape(dim), s_y.reshape(dim)


# class XYPolynomial(Surface):
#     """
#     General XY polynomial surface of equation of parameters:
    
#     explicit:   b z^2 - z + \sum{i,j} a_ij x^i y^{j-i} = 0
#     implicit:   (denote c = \sum{i,j} a_ij x^i y^{j-i})
#                 z = (1 - \sqrt{1 - 4 b c}) / (2b)
                
#     explicit derivatives:
#     (2 b z - 1) dz + \sum{i,j} a_ij x^{i-1} y^{j-i-1} ( i y dx + (j-i) x dy ) = 0

#     dx = \sum{i,j} a_ij   i   x^{i-1} y^{j-i}
#     dy = \sum{i,j} a_ij (j-i) x^{i}   y^{j-i-1}
#     dz = 2 b z - 1
#     """
#     def __init__(self, r, d, J=0, ai=None, b=None, is_square=False, device=torch.device('cpu')):
#         Surface.__init__(self, r, d, is_square, device)
#         self.J  = J
#         # differentiable parameters (default: all ai's and b are zeros)
#         if ai is None:
#             self.ai = torch.zeros(self.J2aisize(J)) if J > 0 else torch.array([0])
#         else:
#             if len(ai) != self.J2aisize(J):
#                 raise Exception("len(ai) != (J+1)*(J+2)/2 !")
#             self.ai = torch.Tensor(ai).to(device)
#         if b is None:
#             b = 0.
#         self.b = torch.Tensor(np.asarray(b)).to(device)
#         print('ai.size = {}'.format(self.ai.shape[0]))
#         self.to(self.device)
    
#     @staticmethod
#     def J2aisize(J):
#         return int((J+1)*(J+2)/2)

#     def center(self):
#         x0 = -self.ai[2]/self.ai[5]
#         y0 = -self.ai[1]/self.ai[3]
#         return x0, y0

#     def fit(self, x, y, z):
#         x, y, z = (torch.Tensor(v.flatten()) for v in [x, y, z])
#         A, AT = self._construct_A(x, y, z**2)
#         coeffs = torch.solve(AT @ z[...,None], AT @ A)[0]
#         self.b  = coeffs[0][0]
#         self.ai = coeffs[1:].flatten()

#     # === Common methods
#     def g(self, x, y):
#         if type(x) is torch.Tensor:
#             c = torch.zeros_like(x)
#         elif type(x) is np.ndarray:
#             c = np.zeros_like(x)
#         else:
#             c = 0.0
#         count = 0
#         for j in range(self.J+1):
#             for i in range(j+1):
#                 # c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
#                 c = c + self.ai[count] * x**i * y**(j-i)
#                 count += 1
#         return c

#     def dgd(self, x, y):
#         if type(x) is torch.Tensor:
#             sx = torch.zeros_like(x)
#             sy = torch.zeros_like(x)
#         elif type(x) is np.ndarray:
#             sx = np.zeros_like(x)
#             sy = np.zeros_like(x)
#         else:
#             sx = 0.0
#             sy = 0.0
#         count = 0
#         for j in range(self.J+1):
#             for i in range(j+1):
#                 if j > 0:
#                     # sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
#                     # sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
#                     sx = sx + self.ai[count] * i * x**max(i-1,0) * y**(j-i)
#                     sy = sy + self.ai[count] * (j-i) * x**i * y**max(j-i-1,0)
#                 count += 1
#         return sx, sy

#     def h(self, z):
#         return self.b * z**2 - z

#     def dhd(self, z):
#         return 2 * self.b * z - torch.ones_like(z)

#     def surface(self, x, y):
#         c = self.g(x, y)
#         return self._solve_for_z(c)

#     def reverse(self):
#         self.b = -self.b
#         self.ai = -self.ai

#     def surface_derivatives(self, x, y):
#         x, y = (v if torch.is_tensor(x) else torch.Tensor(v) for v in [x, y])
#         sx = torch.zeros_like(x)
#         sy = torch.zeros_like(x)
#         c = torch.zeros_like(x)
#         count = 0
#         for j in range(self.J+1):
#             for i in range(j+1):
#                 c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
#                 if j > 0:
#                     sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
#                     sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
#                 count += 1
#         z = self._solve_for_z(c)
#         return sx, sy, self.dhd(z)
        
#     def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z):
#         #pylint: disable=unused-argument
#         # (basically a copy of `surface_derivatives`)
#         x = ox + t * dx
#         y = oy + t * dy
#         sx = torch.zeros_like(x)
#         sy = torch.zeros_like(x)
#         c = torch.zeros_like(x)
#         count = 0
#         for j in range(self.J+1):
#             for i in range(j+1):
#                 c = c + self.ai[count] * torch.pow(x, i) * torch.pow(y, j-i)
#                 if j > 0:
#                     sx = sx + self.ai[count] * i * torch.pow(x, max(i-1,0)) * torch.pow(y, j-i)
#                     sy = sy + self.ai[count] * (j-i) * torch.pow(x, i) * torch.pow(y, max(j-i-1,0))
#                 count += 1
#         s = c + self.h(z)
#         return s, sx*dx + sy*dy + self.dhd(z)*dz

#     # === Private methods
#     def _construct_A(self, x, y, A_init=None):
#         A = torch.zeros_like(x) if A_init == None else A_init
#         for j in range(self.J+1):
#             for i in range(j+1):
#                 A = torch.vstack((A, torch.pow(x, i) * torch.pow(y, j-i)))
#         AT = A[1:,:] if A_init == None else A
#         return AT.T, AT

#     def _solve_for_z(self, c):
#         # TODO: potential NaN grad
#         if self.b == 0:
#             return c
#         else:
#             return (1. - torch.sqrt(1. - 4*self.b*c)) / (2*self.b)


# class Mesh(Surface):
#     """
#     Linear mesh representation for freeform surface.
#     """
#     def __init__(self, r, d, size, c=None, is_square=False, device=torch.device('cpu')):
#         Surface.__init__(self, r, d, is_square, device)
#         if c is None:
#             self.c = torch.zeros(size).to(device)
#         else:
#             c = np.asarray(c)
#             if c.size != np.prod(c_shape):
#                 raise Exception('len(c) is not correct!')
#             self.c = torch.Tensor(c.reshape(*c_shape)).to(device)
#         self.size = torch.Tensor(np.array(size)) # screen image dimension [pixel]
#         self.size_np = size # screen image dimension [pixel]

#     # === Common methods
#     def g(self, x, y):
#         return self._shading(x, y)
    
#     def dgd(self, x, y):
#         p = (torch.stack((x,y), axis=-1)/(2*self.r) + 0.5) * (self.size-1)
#         p_floor = torch.floor(p).long()
#         x0, y0 = p_floor[...,0], p_floor[...,1]
#         s00, s01, s10, s11 = self._tex4(x0, y0)
#         denominator = 2 * (2*self.r / self.size)
#         return (s10 - s00 + s11 - s01) / denominator[0], (s01 - s00 + s11 - s10) / denominator[1]

#     def h(self, z):
#         return -z

#     def dhd(self, z):
#         return -torch.ones_like(z)

#     def surface(self, x, y):
#         return self._shading(x, y)

#     def surface_derivatives(self, x, y):
#         sx, sy = self.dgd(x, y)
#         return sx, sy, -torch.ones_like(x)

#     def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z):
#         #pylint: disable=unused-argument
#         x = ox + t * dx
#         y = oy + t * dy
        
#         p = (torch.stack((x,y), axis=-1)/(2*self.r) + 0.5) * (self.size-1)
#         p_floor = torch.floor(p).long()

#         # linear interpolation
#         x0, y0 = p_floor[...,0], p_floor[...,1]
#         s00, s01, s10, s11 = self._tex4(x0, y0)
#         w1 = p - p_floor
#         w0 = 1. - w1
#         s = (
#             w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
#             w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
#         )
#         denominator = 2 * (2*self.r / (self.size-1))
#         sx = (s10 - s00 + s11 - s01) / denominator[0]
#         sy = (s01 - s00 + s11 - s10) / denominator[1]
#         return s - z, sx*dx + sy*dy - dz

#     def reverse(self):
#         self.c = -self.c

#     # === Private methods
#     def _tex(self, x, y, bmode=BoundaryMode.replicate): # texture indexing function
#             if bmode is BoundaryMode.zero:
#                 raise NotImplementedError()
#             elif bmode is BoundaryMode.replicate:
#                 x = torch.clamp(x, min=0, max=self.size_np[0]-1)
#                 y = torch.clamp(y, min=0, max=self.size_np[1]-1)
#             elif bmode is BoundaryMode.symmetric:
#                 raise NotImplementedError()
#             elif bmode is BoundaryMode.periodic:
#                 raise NotImplementedError()
#             img = self.c[x.flatten(), y.flatten()]
#             return img.reshape(x.shape)

#     def _tex4(self, x0, y0, bmode=BoundaryMode.replicate): # texture indexing four pixels
#         s00 = self._tex(  x0,   y0, bmode)
#         s01 = self._tex(  x0, 1+y0, bmode)
#         s10 = self._tex(1+x0,   y0, bmode)
#         s11 = self._tex(1+x0, 1+y0, bmode)
#         return s00, s01, s10, s11

#     def _shading(self, x, y, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
#         p = (torch.stack((x,y), axis=-1)/(2*self.r) + 0.5) * (self.size-1)
#         p_floor = torch.floor(p).long()

#         if lmode is InterpolationMode.nearest:
#             val = self._tex(p_floor[...,0], p_floor[...,1], bmode)
#         elif lmode is InterpolationMode.linear:
#             x0, y0 = p_floor[...,0], p_floor[...,1]
#             s00, s01, s10, s11 = self._tex4(x0, y0, bmode)
#             w1 = p - p_floor
#             w0 = 1. - w1
#             val = (
#                 w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
#                 w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
#             )
#         return val


# class OuterProduct(Surface):
#     """
#     OuterProduct surface: z(x,y) = fx(x; ax) fy(y; ay).
#     """
#     def __init__(self, r, d, fx, fy, dfx, dfy, ax=None, ay=None, is_square=False, device=torch.device('cpu')):
#         Surface.__init__(self, r, d, is_square, device)
#         self.ax = None
#         self.ay = None
#         if ax is not None:  self.ax = torch.Tensor(np.array(ax))
#         if ay is not None:  self.ay = torch.Tensor(np.array(ay))

#         # function handles
#         self.fx = lambda x: fx(x, self.ax)
#         self.fy = lambda y: fy(y, self.ay)
#         self.dfx = lambda x: dfx(x, self.ax)
#         self.dfy = lambda y: dfy(y, self.ay)
#         self.isforward = 1.0
        
#     # === Common methods
#     def g(self, x, y):
#         # return self.fx(x) * self.fy(y)
#         return self.isforward * (self.fx(x) + self.fy(y))

#     def dgd(self, x, y):
#         # return self.fy(y) * self.dfx(x), self.fx(x) * self.dfy(y)
#         return self.isforward * self.dfx(x), self.isforward * self.dfy(y)

#     def h(self, z):
#         return -z

#     def dhd(self, z):
#         return -torch.ones_like(z)

#     def surface(self, x, y):
#         return self.isforward * self.g(x, y)

#     def reverse(self): # TODO: only valid when you can assume surface(x; -a) = -surface(x; a)
#         self.isforward = -1.0

#     def surface_derivatives(self, x, y):
#         dx, dy = self.dgd(x, y)
#         return dx, dy, -torch.ones_like(x)




class AspheCubic(Surface):
    """ Aspherical + Cubic

        a for aspherical surface, b for cubic phase surface.
    """
    def __init__(self, r, d, c, k, ai=None, b=None, mat2=None, is_square=False, device=DEVICE):
        Surface.__init__(self, r, d, mat2, is_square=is_square, device=device)

        self.c = torch.Tensor([c])
        self.k = torch.Tensor([k])

        if ai is not None:
            self.ai_degree = len(ai)
            self.ai = torch.tensor(ai)
            for i, ai0 in enumerate(ai):
                exec(f'self.ai{2*i+2} = torch.Tensor([{ai0}])')
        else:
            self.ai_degree = 0
            self.ai = None

        if b is not None:
            self.b_degree = len(b) * 2 + 1
            self.b = torch.tensor(b)
            for i, bi in enumerate(b):
                exec(f'self.b{2*i+3} = torch.Tensor([{bi}])')
        else:
            self.b_degree = 0
            self.b = None
        
        self.rotate_angle = 0
        self.to(device)

    def g(self, x, y):
        """ Compute surface height with a mask.
        """
        r2 = x**2 + y**2 

        # Aspherical part 
        z_asp = r2 * self.c / (1 + torch.sqrt(1 - (1 + self.k) * r2 * self.c**2))
        for i in range(1, self.ai_degree+1):
            exec(f'z_asp += self.ai{2*i} * r2 ** {i}')

        # Cubic part, including a rotation to consider assembling error
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 3:
            z_cubic = self.b3 * (x**3 + y**3)
        elif self.b_degree == 5:
            z_cubic = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 7:
            z_cubic = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5) + self.b7 * (x**7 + y**7)
        else:
            raise Exception('Unsupported cubic degree!')

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = - x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        # Combine two parts
        z = z_asp + z_cubic
        
        if len(z.size()) == 0:  # when receive a float (x, y)
            z = torch.tensor([z]).to(self.device)
        
        return z

    def dgd(self, x, y):
        """ Compute surface height derivatives to x and y.
        """
        r2 = x**2 + y**2

        # Aspherical part
        sf = torch.sqrt(1-(1+self.k)*r2*self.c**2)
        dsdr2 = (1+sf+(1+self.k)*r2*self.c**2/2/sf) *self.c/(1+sf)**2
        for i in range(1, self.ai_degree+1):
            exec(f'dsdr2 += {i} * self.ai{2*i} * r2 ** {i-1}')

        # Cubic part
        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) - y * np.sin(self.rotate_angle)
            y = x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        if self.b_degree == 3:
            sx_cubic = 3 * self.b3 * x**2
            sy_cubic = 3 * self.b3 * y**2
        elif self.b_degree == 5:
            sx_cubic = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            sy_cubic = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 7:
            sx_cubic = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            sy_cubic = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise Exception('Unsupported cubic degree!')

        if self.rotate_angle != 0:
            x = x * np.cos(self.rotate_angle) + y * np.sin(self.rotate_angle)
            y = -x * np.sin(self.rotate_angle) + y * np.cos(self.rotate_angle)

        # Combine
        sx = dsdr2 * 2 * x + sx_cubic
        sy = dsdr2 * 2 * y + sy_cubic

        return sx, sy

    def max_height(self):
        """ Maximum valid height.
        """
        if self.k > -1:
            max_height = torch.sqrt(1/(self.k+1)/(self.c**2)).item() - 0.01
        else:
            # always valid, we can set it to a large value.
            max_height = 100

        return max_height

    def perturb(self, curvature_precision=0.001, thickness_precision=0.0005, diameter_precision=0.01, angle=0.01):
        """ Perturb the surface
        """
        # Perturb aspherical part
        self.r += np.random.randn() * diameter_precision
        if self.c != 0:
            self.c *= 1 + np.random.randn() * curvature_precision
        if self.d != 0:
            self.d += np.random.randn() * thickness_precision
        if self.k != 0:
            self.k *= 1 + np.random.randn() * curvature_precision
        for i in range(1, self.ai_degree+1):
            exec(f'self.ai{2*i} *= 1 + np.random.randn() * curvature_precision')
        
        # Perturb cubic part
        if self.b_degree == 3:
            self.b3 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 5:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
        elif self.b_degree == 7:
            self.b3 *= 1 + np.random.randn() * curvature_precision
            self.b5 *= 1 + np.random.randn() * curvature_precision
            self.b7 *= 1 + np.random.randn() * curvature_precision

        self.rotate_angle = np.random.randn() * angle

    
    def activate_grad(self, activate_ai=True, activate_b=True, activate_d=True):
        """ Activate/deactivate greadients.
        """
        self.d.requires_grad_(activate_d)
        
        if activate_b:
            if self.b_degree == 3:
                self.b3.requires_grad_(True)
            elif self.b_degree == 5:
                self.b3.requires_grad_(True)
                self.b5.requires_grad_(True)
            elif self.b_degree == 7:
                self.b3.requires_grad_(True)
                self.b5.requires_grad_(True)
                self.b7.requires_grad_(True)
            else:
                raise Exception('Unimplemented.')

        if activate_ai:
            if self.ai_degree == 4:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
            elif self.ai_degree == 6:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
            elif self.ai_degree == 8:
                self.ai2.requires_grad_(True)
                self.ai4.requires_grad_(True)
                self.ai6.requires_grad_(True)
                self.ai8.requires_grad_(True)
            else:
                raise Exception('Unimplemented.')