""" Geometry optics. Lensgroup implementation.
"""
import torch
import glob
import random
import json
import cv2 as cv
import warnings
import statistics
from tqdm import tqdm
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from transformers import get_cosine_schedule_with_warmup

from .surfaces import *
from .utils import *
from .basics import GEO_SPP, EPSILON, WAVE_RGB

class Lensgroup():
    """
    The Lensgroup (consisted of multiple optical surfaces) is mounted on a rod, whose
    origin is `origin`. The Lensgroup has full degree-of-freedom to rotate around the
    x/y axes, with the rotation angles defined as `theta_x`, `theta_y`, and `theta_z` (in degree).

    In Lensgroup's coordinate (i.e. object frame coordinate), surfaces are allocated
    starting from `z = 0`. There is an additional, comparatively small 3D origin shift
    (`shift`) between the surface center (0,0,0) and the origin of the mount, i.e.
    shift + origin = lensgroup_origin.
    
    There are two configurations of ray tracing: forward and backward. In forward mode,
    rays start from `d = 0` surface and propagate along the +z axis; In backward mode,
    rays start from `d = d_max` surface and propagate along the -z axis.

    Elements:
    ``` == Varaiables                               Requires gradient
        origin [Tensor]:                            ?
        shift [Tensor]:                             ?
        theta_x [Tensor]:                           ?
        theta_y [Tensor]:                           ?
        theta_z [Tensor]:                           ?
        to_world [Tranformation]:                   ?
        to_object [Transformation]:                 ?
        surfaces (list):                            True
            r                                       float
            d                                       tensor
            c                                       tensor
            k                                       tensor
            ai                                      tensor list
        materials (list):                           

        == Float/Int:
        aper_idx: aperture index.                   False
        foclen                                      False
        fnum                                        False
        fov(half diagonal fov)                      False
        imgh(diagonal sensor distance)              False
        sensor:
            sensor_size
            sensor_res
            pixel_size
            r_last(half diagonal distance)
            d_sensor
            focz

        == String/Boolean:
        lens_name [string]:                         False
        device [string]: 'cpu' or 'cuda:0'          False
        # mts_prepared [Bool]:                      False
        sensor_prepared [Bool]:                     False
        
    ```

    Methods (divided into some groups):
    ```
        Init
        Ray sampling
        Ray tracing
        Ray-tracing-based rendering
        PSF
        Geometrical optics (calculation)
        Lens operation
        Visualization
        Calibration and distortion
        Loss function
        Optimization
        Lens field IO
        Others
    ```

    """
    def __init__(self, filename=None, origin=np.zeros(3), shift=np.zeros(3), theta_x=0., theta_y=0., theta_z=0., 
                 sensor_res=(1024, 1024), use_roc=False, post_computation=True, filter=False, device=torch.device('cuda')):
        """ Initialize Lensgroup.

        Args:
            filename (string): lens file.
            origin (1*3 array): center point.
            shift (1*3 array): center shift.
            theta_x (): Can be replaced by 1*3 array???
            theta_y (): 
            theta_z (): 
            device ('cpu' or 'cuda'): We need to spercify device here, because `sample_ray` needs it.
            sensor_res: (H, W)
        """
        super(Lensgroup, self).__init__()
        self.origin = torch.Tensor(origin)
        self.shift = torch.Tensor(shift)
        self.theta_x = torch.Tensor(np.asarray(theta_x))
        self.theta_y = torch.Tensor(np.asarray(theta_y))
        self.theta_z = torch.Tensor(np.asarray(theta_z))
        self.filter = filter
        self.device = device

        # Load lens file.
        if filename is not None:
            self.lens_name = filename
            self.load_file(filename, use_roc, post_computation, sensor_res)
        
        # Move all variables and sub-classes to device.
        self.to(device)


    def to(self, device=torch.device('cuda')):
        """ Move all variables to target device.
        """
        for key, val in vars(self).items():
            if torch.is_tensor(val):
                exec('self.{x} = self.{x}.to(device)'.format(x=key))
            elif val.__class__.__name__ in ('list', 'tuple'):
                for i, v in enumerate(val):
                    if torch.is_tensor(v):
                        exec('self.{x}[{i}] = self.{x}[{i}].to(device)'.format(x=key, i=i))
        
        self.device = device
        return self


    def load_file(self, filename, use_roc, post_computation, sensor_res):
        """ Load lens from .txt file.

        Args:
            filename (string): lens file.
            use_roc (bool): use radius of curvature (roc) or not. In the old code, we store lens data in roc rather than curvature.
            post_computation (bool): compute fnum, fov, foclen or not.
            sensor_res (list): sensor resolution.
        """
        if filename[-4:] == '.txt':
            self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(filename, use_roc)
            
            self.d_sensor = d_last + self.surfaces[-1].d.item()
            self.focz = self.d_sensor

            self.find_aperture()
            self.prepare_sensor(sensor_res)

            if post_computation:
                self.post_computation()

            if self.filter:
                self.surfaces[-1].square = True
                self.surfaces[-2].square = True

        elif filename[-5:] == '.json':
            self.sensor_res = sensor_res
            self.read_lens_json(filename)

        else:
            raise Exception('Unknown file type.')
        

    def load_external(self, surfaces, materials, r_last, d_sensor):
        """ Load lens from extrenal surface/material list.
        """
        self.surfaces = surfaces
        self.materials = materials
        self.r_last = r_last
        self.d_sensor = d_sensor
        

    def prepare_sensor(self, sensor_res=(512, 512)):
        """ Create sensor. 

            reference values:
                Nikon z35 f1.8: diameter = 1.912 [cm] ==> But can we just use [mm] in our code?
                Congli's caustic example: diameter = 12.7 [mm]
        Args:
            sensor_res (list): Resolution, pixel number.
            pixel_size (float): Pixel size in [mm].

            sensor_res: (H, W)
        """
        H, W = sensor_res
        self.sensor_res = (H, W)
        self.sensor_size = [2 * self.r_last * H / np.sqrt(H**2 + W**2), 2 * self.r_last * W / np.sqrt(H**2 + W**2)]
        self.pixel_size = self.sensor_size[0] / sensor_res[0]


    def post_computation(self):
        """ After loading lens, compute foclen, fov and fnum.
        """
        self.find_aperture()
        self.hfov = self.calc_fov()
        self.foclen = self.calc_efl()
        
        if self.aper_idx is not None:
            avg_pupilz, avg_pupilx = self.entrance_pupil()
            self.fnum = self.foclen / avg_pupilx / 2


    def find_aperture(self):
        """ Find aperture by last and next material.
        """
        self.aper_idx = None
        for i in range(len(self.surfaces)-1):
            if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # AIR or OCCLUDER
                self.aper_idx = i
                return

        # if not found, use the narrowest surface (Now we are using the first surface)
        if self.aper_idx is None:
            print("There is no aperture stop in the lens, use the first surface as the aperture.")


    def find_diff_surf(self):
        """ Get surface indices without aperture.
        """
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(range(self.aper_idx+1, len(self.surfaces)))
        return diff_surf_range




    # ====================================================================================
    # Ray Sampling
    # ====================================================================================

    @torch.no_grad()
    def sample_parallel_2D(self, R=None, wavelength=DEFAULT_WAVE, z=None, view=0.0, M=15, forward=True, entrance_pupil=False):
        """ Sample parallel 2D rays on x-axis. Used for fast 2D geometry optics. Ray shape [M, 3]

        Args:
            R (float, optional): sampling radius. Defaults to None.
            wavelength (float, optional): ray wavelength. Defaults to DEFAULT_WAVE.
            z (float, optional): sampling depth. Defaults to None.
            view (float, optional): incident angle (in degree). Defaults to 0.0.
            M (int, optional): ray number. Defaults to 15.
            forward (bool, optional): forward or backward rays. Defaults to True.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
        """
        if entrance_pupil:
            pupilz, pupilx = self.entrance_pupil()
            x2 = torch.linspace(-pupilx, pupilx, M, device=self.device) * 0.99
            y2 = torch.zeros_like(x2)
            z2 = torch.full_like(x2, pupilz)
            o2 = torch.stack((x2,y2,z2), axis=1)
            
            dx = torch.full_like(x2, np.tan(view/57.3))
            dy = torch.zeros_like(x2)
            dz = torch.full_like(x2, 1)
            d = torch.stack((dx,dy,dz), axis=1)

            if pupilz > 0:
                o = o2 - d * (z2 / dz).unsqueeze(-1)
            else:
                o = o2

            return Ray(o, d, wavelength, device=self.device, normalized=False)
        
        else:
            # ray origin
            x = torch.linspace(-R, R, M, device=self.device)
            y = torch.zeros_like(x)
            if z is None:
                z = 0 if forward else self.surfaces[-1].d+0.5
            z = torch.full_like(x, z)
            o = torch.stack((x,y,z), axis=1)
            
            # ray direction
            angle = torch.Tensor(np.asarray(np.radians(view))).to(self.device)
            ones = torch.ones_like(x)
            if forward:
                d = torch.stack((
                    torch.sin(angle)*ones,
                    torch.zeros_like(ones),
                    torch.cos(angle)*ones), axis=-1
                )
            else:
                d = torch.stack((
                    torch.sin(angle)*ones,
                    torch.zeros_like(ones),
                    -torch.cos(angle)*ones), axis=-1
                )

            return Ray(o, d, wavelength=wavelength, device=self.device)


    @torch.no_grad()
    def sample_parallel(self, wavelength=DEFAULT_WAVE, fov=0.0, R=None, z=0., M=15, sampling='grid', forward=True, entrance_pupil=False):
        """ Sample parallel rays from plane (-R:R, -R:R, z). This function is usually called to get plane incident rays , for example refocusing.

        Usage:
        ```
            if sample on a given 2D circle plane, set R;
            if sample in an angle range, set fov;
        ```

        Args:
            wavelength (float, optional): ray wavelength. Defaults to DEFAULT_WAVE.
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            R (float, optional): sampling radius. Defaults to None.
            z (float, optional): sampling depth. Defaults to 0..
            M (int, optional): ray number. Defaults to 15.
            sampling (str, optional): sampling method. Defaults to 'grid'.
            forward (bool, optional): forward or backward rays. Defaults to True.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [spp, M, M]
        """
        fov = np.radians(np.asarray(fov))   # convert degree to radian
        
        # ==> Sample rays
        if entrance_pupil:
            pupilz, pupilr = self.entrance_pupil()
            if sampling == 'grid':
                # sample a square larger than pupil circle
                x, y = torch.meshgrid(
                    torch.linspace(-pupilr, pupilr, M),
                    torch.linspace(pupilr, -pupilr, M),
                    indexing='xy'
                )
            elif sampling == 'radial':
                r2 = torch.rand((M, M)) * pupilr**2
                theta = torch.rand((M, M)) * 2 * np.pi
                x = torch.sqrt(r2) * torch.cos(theta)
                y = torch.sqrt(r2) * torch.sin(theta)

        else:
            if R is None:
                # We want to sample at a depth, so radius of the cone need to be computed.
                with torch.no_grad():
                    sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0).item() # sag is a float
                    R = np.tan(fov) * sag + self.surfaces[0].r

            if sampling == 'grid':
                raise Exception('Check meshgrid function!')
                
            elif sampling == 'radial':
                r = torch.linspace(0, R, M)
                theta = torch.linspace(0, 2*np.pi, M+1)[0:M]
                x = r[None,...] * torch.cos(theta[...,None])
                y = r[None,...] * torch.sin(theta[...,None])


        # ==> Generate rays
        if isinstance(fov, float):
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2)
            d = torch.zeros_like(o)
            if forward:
                d[...,2] = torch.full_like(x, np.cos(fov))
                d[...,0] = torch.full_like(x, np.sin(fov))
            else:
                d[...,2] = torch.full_like(x, -np.cos(fov))
                d[...,0] = torch.full_like(x, -np.sin(fov))
        else:
            spp = len(fov)
            o = torch.stack((x, y, torch.full_like(x, pupilz)), axis=2).unsqueeze(0).repeat(spp, 1, 1, 1)
            d = torch.zeros_like(o)
            for i in range(spp):
                if forward:
                    d[i, :, :, 2] = torch.full_like(x, np.cos(fov[i]))
                    d[i, :, :, 0] = torch.full_like(x, np.sin(fov[i]))
                else:
                    d[i, :, :, 2] = torch.full_like(x, -np.cos(fov[i]))
                    d[i, :, :, 0] = torch.full_like(x, -np.sin(fov[i]))

        rays = Ray(o, d, wavelength, device=self.device)
        rays.propagate_to(z)
        return rays


    @torch.no_grad()
    def sample_point_source_2D(self, depth=-1000, view=0, M=9, entrance_pupil=False, wavelength=DEFAULT_WAVE):
        """ Sample point source 2D rays in x-axis. Ray shape [M, 3].

            Currently used for 2D drawing.

        Args:
            depth (float, optional): sampling depth. Defaults to -1000.
            view (float, optional): incident angle (in degree). Defaults to 0.
            M (int, optional): ray number. Defaults to 9.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.
            wavelength (float, optional): ray wavelength. Defaults to DEFAULT_WAVE.
        """
        if entrance_pupil:
            pupilz, pupilx = self.entrance_pupil()
        else:
            pupilz, pupilx = 0, self.surfaces[0].r

        # Second point on the pupil or first surface
        x2 = torch.linspace(-pupilx, pupilx, M, device=self.device) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2,y2,z2), axis=1)

        # First point is the point source
        o1 = torch.zeros_like(o2)
        o1[:, 2] = depth
        o1[:,0] = depth * np.tan(view / 57.3)

        # Form the rays and propagate to z = 0
        d = o2 - o1
        ray = Ray(o1, d, wavelength=wavelength, device=self.device, normalized=False)
        ray.propagate_to(z=0)

        return ray


    @torch.no_grad()
    def sample_point_source(self, R=None, depth=-10.0, spp=16, fov=10.0, M=11, forward=True, shrink=False, pupil=False, wavelength=DEFAULT_WAVE, importance_sampling=False):
        """ Sample 3D point source rays. Rays come from a 2D square array (-R~R, -Rw~Rw, depth), and fall into a cone   spercified by fov or pupil. Rays have shape [spp, M, M, 3]

        Args:
            R (float, optional): sample plane half side length. Defaults to None.
            depth (float, optional): sample plane z position. Defaults to -10.0.
            spp (int, optional): sample per pixel. Defaults to 16.
            fov (float, optional): cone angle. Defaults to 10.0.
            M (int, optional): sample plane resolution. Defaults to 11.
            forward (bool, optional): forward or backward rays. Defaults to True.
            shrink (bool, optional): whether to shrink the sample plane. Defaults to False.
            pupil (bool, optional): whether to use pupil. Defaults to False.
            wavelength (float, optional): ray wavelength. Defaults to DEFAULT_WAVE.
        """
        if R is None:
            R = self.surfaces[0].r

        Rw = R * self.sensor_res[1] / self.sensor_res[0] # half height

        # sample o
        if shrink:
            raise Exception('Should abandon this operation.')
        else:
            x, y = torch.meshgrid(
                torch.linspace(-1, 1, M),
                torch.linspace(1, -1, M),
                indexing='xy'
                )

        if importance_sampling:
            x = torch.sqrt(x.abs()) * x.sign()
            y = torch.sqrt(y.abs()) * y.sign()

        x = x * Rw
        y = y * R

        z = torch.full_like(x, depth)
        o = torch.stack((x,y,z), -1).to(self.device)
        o = o.unsqueeze(0).repeat(spp, 1, 1, 1)
        
        # sample d
        if pupil:
            o2 = self.sample_pupil(res=(M,M), spp=spp)
            d = o2 - o
            d = d / torch.linalg.vector_norm(d, ord=2, dim=-1, keepdim=True)
        else:
            # https://en.wikipedia.org/wiki/Spherical_coordinate_system
            # cone shape rays
            theta = torch.deg2rad(torch.rand(spp, M, M) * fov).to(self.device)    # [0, fov]
            phi = (torch.rand(spp, M, M)*2*np.pi).to(self.device)   # [0, 2pi]
            if forward:
                d = torch.stack((
                    torch.sin(theta)*torch.cos(phi),
                    torch.sin(theta)*torch.sin(phi),
                    torch.cos(theta)
                ), axis=-1)
            else:
                d = torch.stack((
                    torch.sin(theta)*torch.cos(phi),
                    torch.sin(theta)*torch.sin(phi),
                    -torch.cos(theta)
                ), axis=-1)

        # generate ray
        ray = Ray(o, d, wavelength=wavelength, device=self.device)
        return ray


    @torch.no_grad()
    def sample_from_points(self, o=[[0, 0, -10000]], spp=8, forward=True, pupil=True, fov=10, wavelength=DEFAULT_WAVE):
        """ Sample rays from given 3D point source. Rays fall into a cone spercified by fov or pupil.
            
            o shape [N, 3], Ray shape [spp, N].

        Args:
            o (list): ray origin. Defaults to [[0, 0, -10000]].
            spp (int): sample per pixel. Defaults to 8.
            forward (bool): forward or backward rays. Defaults to True.
            pupil (bool): whether to use pupil. Defaults to True.
            fov (float): cone angle. Defaults to 10.
            wavelength (float): ray wavelength. Defaults to DEFAULT_WAVE.
        """
        # ==> Compute o, shape [spp, N, 3]
        if not torch.is_tensor(o):
            o = torch.tensor(o)
        o = o.unsqueeze(0).repeat(spp, 1, 1)
        
        # ==> Sample pupil and compute d
        pupilz, pupilr = self.entrance_pupil()
        theta = torch.rand(spp)*2*np.pi
        r = torch.sqrt(torch.rand(spp)*pupilr**2)
        x2 = r * torch.cos(theta)
        y2 = r * torch.sin(theta)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2,y2,z2), 1)
        
        d = o2.unsqueeze(1) - o
        d = d / torch.linalg.vector_norm(d, ord=2, dim=-1, keepdim=True)
        
        # ==> Generate ray
        ray = Ray(o, d, wavelength=wavelength, device=self.device)
        return ray


    @torch.no_grad()
    def sample_pupil(self, res=(512,512), spp=16, num_angle=8, pupilr=None, pupilz=None, multiplexing=False):
        """ Sample points on the pupil plane with rings. Can only sample points, not rays. Points have shape [spp, res, res].

            2*pi is devided into [num_angle] sectors.
            Circle is devided into [spp//num_angle] rings.

        Args:
            res (tuple): pupil plane resolution. Defaults to (512,512).
            spp (int): sample per pixel. Defaults to 16.
            num_angle (int): number of sectors. Defaults to 8.
            pupilr (float): pupil radius. Defaults to None.
            pupilz (float): pupil z position. Defaults to None.
            multiplexing (bool): whether to use multiplexing. Defaults to False.
        """
        H, W = res
        if pupilr is None or pupilz is None:
            pupilz, pupilr = self.entrance_pupil()

        # => Naive implementation
        if spp % num_angle != 0 or spp >= 10000:
            theta = torch.rand((spp, H, W), device=self.device) * 2 * np.pi
            r2 = torch.rand((spp, H, W), device=self.device) * pupilr**2
            r = torch.sqrt(r2)

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x,y,z), -1)

        # => Sample more uniformly
        else:
            num_r2 = spp // num_angle
            x, y = [], []
            for i in range(num_angle):
                for j in range(spp//num_angle):
                    delta_theta = torch.rand((1, *res), device=self.device) * 2 * np.pi / num_angle # sample delta_theta from [0, pi/4)
                    theta = delta_theta + i * 2 * np.pi / num_angle 

                    delta_r2 = torch.rand((1, *res), device=self.device) * pupilr**2 / spp * num_angle
                    r2 = delta_r2 + j * pupilr**2 / spp * num_angle
                    r = torch.sqrt(r2)

                    x.append(r * torch.cos(theta))
                    y.append(r * torch.sin(theta))
            
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            z = torch.full_like(x, pupilz)
            o = torch.stack((x,y,z), -1)

        return o




    # ====================================================================================
    # Ray Tracing functions
    # ====================================================================================

    def trace(self, ray, stop_ind=None, lens_range=None, ignore_aper=False, record=False):
        """ Ray tracing function. Ray in and ray out.

            Forward or backward ray tracing is automatically determined by ray directions.

        Args:
            ray (Ray object): Ray object.
            stop_ind (int): Early stop index.
            lens_range (list): Lens range to trace.
            ignore_aper (bool): Whether to ignore aperture stop.
            record (bool): Whether to record ray path.

        Returns:
            ray_out (Ray object): ray after optical system.
            valid (boolean matrix): mask denoting valid rays.
            oss (): positions of ray.
        """
        # Early stop. If None, stop between last lens surface and sensor.
        if stop_ind is None:
            stop_ind = len(self.surfaces) - 1
        
        if lens_range is None:
            lens_range = range(0, len(self.surfaces))
        
        if ignore_aper:
            lens_range = self.find_diff_surf()

        # Determine forward or backward
        with torch.no_grad():
            is_forward = ray.d.reshape(-1,3)[0,2]>0

        if is_forward:
            valid, ray_out, oss = self._forward_tracing(ray, lens_range, record=record)
        else:
            valid, ray_out, oss = self._backward_tracing(ray, lens_range, record=record)

        return ray_out, valid, oss


    def trace2obj(self, ray, depth=DEPTH, ignore_aper=False):
        """ Trace rays through the lens and reach the sensor plane.
        """
        ray, _, _, = self.trace(ray, ignore_aper=ignore_aper)
        ray = ray.propagate_to(depth)
        return ray

    
    def trace2sensor(self, ray, ignore_aper=False, record=False):
        """ Trace rays through the lens and reach the sensor plane.

        Args:
            ray: Ray object.
            ignore_aper: Whether to ignore aperture stop.
            record: Whether to record ray path.

        Returns:
            ray: Ray object.
        """
        ray, _, _, = self.trace(ray, ignore_aper=ignore_aper)
        ray = ray.propagate_to(self.d_sensor)
        return ray


    def trace_to_sensor(self, ray, ignore_invalid=False, record=False, ignore_aper=False):
        """ Trace rays form outside object and compute points of intersections with sensor plane.

        Args:   
            ray: Ray object.
            ignore_invalid: Whether to ignore invalid rays.
            record: Whether to record ray path.
            ignore_aper: Whether to ignore aperture stop.

        Returns:
            p: Points of intersections with sensor plane.
        """
        self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0, device=self.device))
        self.materials.append(Material('air'))

        # trace rays through lens system
        lens_range = list(range(len(self.surfaces)))
        if ignore_aper and self.aper_idx is not None:
            lens_range.remove(self.aper_idx)

        ray_out, valid, oss = self.trace(ray=ray, lens_range=lens_range, record=record)
        self.surfaces.pop()
        self.materials.pop()
        p = ray_out.o

        if ignore_invalid:
            p = p[valid]
        else:
            if len(p.shape) < 2:
                return p
            p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))

        if record:
            for v, os, pp in zip(valid, oss, p):
                if v:
                    os.append(pp.cpu().detach().numpy())
            return p, oss
        else:
            return p


    def _forward_tracing(self, ray, lens_range, record):
        """ Trace rays from object to sensor.

        Args:
            ray: Ray object.
            lens_range: Lens range to trace.
            record: Whether to record ray path.

        Returns:
            valid: Mask denoting valid rays.
            ray: Ray object.
            oss: Positions of ray.
        """
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape
        if record:
            oss = []    # oss records all points of intersection
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])
        else:
            oss = None

        for i in lens_range:
            eta = self.materials[i].ior(wavelength) / self.materials[i+1].ior(wavelength)
            ray = self.surfaces[i].ray_reaction(ray, eta)
            
            valid = (ray.ra == 1)
            if record: 
                p = ray.o
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v.any():
                        os.append(pp)
        
        return valid, ray, oss


    def _backward_tracing(self, ray, lens_range, record):
        """ Trace rays from sensor to object.

        Args:   
            ray: Ray object.
            lens_range: Lens range to trace.
            record: Whether to record ray path.

        Returns:
            valid: Mask denoting valid rays.
            ray: Ray object.
            oss: Positions of ray.
        """
        wavelength = ray.wavelength 
        dim = ray.o[..., 2].shape
        valid = (ray.ra == 1)
        
        if record:
            oss = []    # oss records all points of intersection
            for i in range(dim[0]):
                oss.append([ray.o[i,:].cpu().detach().numpy()])
        else:
            oss = None

        for i in np.flip(lens_range):
            eta = self.materials[i+1].ior(wavelength) / self.materials[i].ior(wavelength)
            ray = self.surfaces[i].ray_reaction(ray, eta)

            valid = (ray.ra > 0)
            if record: 
                p = ray.o
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v.any():
                        os.append(pp)

        valid = (ray.ra == 1)
        return valid, ray, oss


        
    # ====================================================================================
    # Ray-tracing based rendering
    # ====================================================================================
    


    # ====================================================================================
    # PSF and spot diagram
    #   1. Incoherent functions
    #   2. Coherent functions
    # ====================================================================================

    # ----------------------------------------------
    # 1. Incoherent PSF functions
    # ----------------------------------------------
    @torch.no_grad()
    def point_source_grid(self, depth, grid=9, normalized=True, quater=False, center=False):
        """ Compute point grid in the object space to compute PSF grid.

        Args:
            depth (float): Depth of the point source plane.
            grid (int): Grid size. Defaults to 9.
            normalized (bool): Whether to use normalized x, y corrdinates [-1, 1]. Defaults to True.
            quater (bool): Whether to use quater of the grid. Defaults to False.
            center (bool): Whether to use center of each patch. Defaults to False.

        Returns:
            point_source: Shape of [grid, grid, 3].
        """
        # ==> Use center of each patch
        if center:
            half_bin_size = 1 / 2 / (grid - 1)
            x, y = torch.meshgrid(
                torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid), 
                torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid),
                indexing='xy')
        # ==> Use corner
        else:   
            x, y = torch.meshgrid(
                torch.linspace(-0.98, 0.98, grid), 
                torch.linspace(0.98, -0.98, grid),
                indexing='xy')
        
        z = torch.full((grid, grid), depth)
        point_source = torch.stack([x, y, z], dim=-1)
        
        # ==> Use quater of the sensor plane to save memory
        if quater:
            z = torch.full((grid, grid), depth)
            point_source = torch.stack([x, y, z], dim=-1)
            bound_i = grid // 2 if grid % 2 == 0 else grid // 2 + 1
            bound_j = grid // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        if not normalized:
            scale = self.calc_scale_pinhole(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source

    
    @torch.no_grad()
    def point_source_radial(self, depth, grid=9, center=False):
        """ Compute point radial in the object space to compute PSF grid.

        Args:
            grid (int, optional): Grid size. Defaults to 9.

        Returns:
            point_source: Shape of [grid, 3].
        """
        # Select center of bin to calculate PSF
        if center:
            half_bin_size = 1 / 2 / (grid - 1)
            x = torch.linspace(0, 1 - half_bin_size, grid)
        else:   
            x = torch.linspace(0, 0.98, grid)
        
        z = torch.full_like(x, depth)
        point_source = torch.stack([x, x, z], dim=-1)
        return point_source


    @torch.no_grad()
    def psf_map(self, M=7, kernel_size=11, depth=DEPTH, spp=2048):
        """ Compute PSF kernels of the lens. return (xc, yc, kernel)

            Shot rays from grid points in object space, trace to sensor and visualize.

            (x, y) coordinates are flipped so that we can directly use it for convolution.
        """
        mag = self.calc_magnification3(depth)

        # ray tracing
        ray = self.sample_point_source(M=M, R=self.sensor_size[0]/2/mag, depth=depth, spp=spp, pupil=True)
        o1 = ray.o.clone()[0,...].cpu()
        ray, _, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        ra = torch.flip(ray.ra.clone().unsqueeze(-1), [1, 2]).cpu()
        o2 = torch.flip(ray.o.clone().cpu(), [1, 2]) * ra
        
        # psf center coordinate
        o2_center = o2.sum(0)/ra.sum(0).add(0.0001)
        xy_center = o2_center[...,:2]
        xy_center_ref = o1[...,:2] * mag
        
        # psf kernel
        xy_norm = (o2[...,:2] - xy_center_ref) * ra # invalid rays are counted as (0, 0)
        x_idx, y_idx = xy_norm[...,0]/self.pixel_size, xy_norm[...,1]/self.pixel_size
        x_idx, y_idx = x_idx.round(), y_idx.round()

        # Use a 'for' loop
        kernel = torch.zeros((M, M, kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[..., i, j] = torch.sum((x_idx == (-kernel_size//2+1+j)) & (y_idx == (kernel_size//2-i)), dim=0) 

        # remove invalid rays contributions to cener kernel
        kernel[:, :, kernel_size//2, kernel_size//2] -= torch.sum(ra!=1, dim=0).squeeze(-1)
        kernel = kernel/torch.sum(kernel, dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1)
        
        return xy_center, kernel


    def assign_points_to_pixels(self, points, ks, x_range, y_range, ra):
        """ Assign points into pixels using bilinear interpolation. The closer a point is to a pixel, the higher impact it makes to this pixel.
        """        
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Normalize points to the range [0, 1]
        points_normalized = torch.zeros_like(points)
        points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
        points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)

        # ==> weight. The thick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * (ks - 1)
        w_b = pixel_indices_float[..., 0] - pixel_indices_float[..., 0].floor()
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()

        # ==> 
        pixel_indices_tl = pixel_indices_float.floor().long()
        pixel_indices_tr = torch.stack((pixel_indices_float[:, 0], pixel_indices_float[:, 1]+1), dim=-1).floor().long()
        pixel_indices_bl = torch.stack((pixel_indices_float[:, 0]+1, pixel_indices_float[:, 1]), dim=-1).floor().long()
        pixel_indices_br = pixel_indices_tl + 1

        # ==>
        grid = torch.zeros(ks, ks).to(points.device)

        # Use advanced indexing to increment the count for each corresponding pixel
        grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra, accumulate=True)
        grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra, accumulate=True)
        grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra, accumulate=True)
        grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra, accumulate=True)

        grid = grid / (grid.sum() + EPSILON)
        grid = grid.view(ks, ks)
        return grid


    def psf_diff(self, point, wvln=DEFAULT_WAVE, kernel_size=21, spp=4096, center=True):
        """ Given point source [spp, 3], output [spp, ks, ks] psf kernel of this point source.

            Incohrent Monte Carlo ray tracing.

            Now this function only supports 1D single-wavelength PSF.

        Args:
            o (Tnesor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. For diff ray tracing, usually kernel_size^2. Defaults to 2048.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            kernel: Shape of [N, ks, ks].
        """
        wvln = wvln * 1e3 if wvln < 10 else wvln    # convert [um] to [nm]
        spp = 16*kernel_size**2 if kernel_size < 32 else spp
        depth = point[2]
        scale = self.calc_scale_pinhole(depth)
        x = self.sensor_size[1] / 2 * scale * point[0]
        y = self.sensor_size[0] / 2 * scale * point[1]
        o = torch.stack((x, y, depth), dim=-1)
        pointc_ref = o[..., :2] / scale

        # sample and trace rays
        ray = self.sample_from_points(o=o, spp=spp, wavelength=wvln)
        ray, _, _ = self.trace(ray)
        ray.prop_to(self.d_sensor)
        ra = ray.ra.unsqueeze(1) # shape [spp, 1, 1]

        # points center, including a flip operation
        points = - ray.o[..., :2]  # shape [spp, 1, 2]
        pointc = torch.sum(points * ra, 0) / ra.sum(0).add(EPSILON)
        psf_range = [(-kernel_size/2+0.5)*self.pixel_size, (kernel_size/2-0.5)*self.pixel_size]
        
        if center:
            points_shift = points - pointc 
        else:
            points_shift = points - pointc_ref.to(self.device)
            
        ra = ra * (points_shift[...,0].abs() < psf_range[1]).unsqueeze(1) * (points_shift[...,1].abs() < psf_range[1]).unsqueeze(1)
        points_shift *= ra
        
        # => PSF
        points_shift = points_shift[:,0,:]
        ra = ra[:,0,0]
        psf = self.assign_points_to_pixels(points=points_shift, ks=kernel_size, x_range=psf_range, y_range=psf_range, ra=ra)

        return psf


    def psf_diff_new(self, point, wvln=DEFAULT_WAVE, kernel_size=21, spp=512, center=True):
        """ Given point source [N, 3], output [N, ks, ks] psf kernel.

            This function is slower than self.psf() because we are using 'for' loop.

        Args:
            o (Tnesor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. Defaults to 1024.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            kernel: Shape of [N, ks, ks].
        """
        depth = point[:, 2]
        scale = self.calc_scale_pinhole(depth)
        x = self.sensor_size[0] / 2 * scale * point[:, 0]
        y = self.sensor_size[1] / 2 * scale * point[:, 1]
        o = torch.stack((x, y, depth), dim=-1)
        pointc_ref = torch.stack((x/scale, y/scale), dim=-1)    # shape [N, 2]

        # sample and trace rays
        ray = self.sample_from_points(o=o, spp=spp, wavelength=wvln)
        ray, _, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        ra = ray.ra.unsqueeze(-1) # shape [spp, N, 1]

        # points center, including a flip operation
        points = - ray.o[..., :2]  # shape [spp, N, 2]
        pointc = torch.sum(points * ra, 0) / ra.sum(0).add(EPSILON)
        psf_range = [(-kernel_size/2+0.5)*self.pixel_size, (kernel_size/2-0.5)*self.pixel_size]
        
        if center:
            points_shift = points - pointc 
        else:
            points_shift = points - pointc_ref.to(self.device)
        
        ra = ra * (points_shift[...,0].abs() < psf_range[1]).unsqueeze(-1) * (points_shift[...,1].abs() < psf_range[1]).unsqueeze(-1)
        points_shift *= ra  # shape [spp, N, 2]
        
        # => PSF
        psfs = []
        for i in range(point.shape[0]): # point shape of [N, 3]
            psf = self.assign_points_to_pixels(points=points_shift[:, i, :], ks=kernel_size, x_range=psf_range, y_range=psf_range, ra=ra[:, i, 0])
            psfs.append(psf)

        psf = torch.stack(psfs, dim=0)
        return psf
    

    def psf_diff_color(self, o, kernel_size=31, spp=4096, center=True):
        """ Compute RGB PSF. Each channel sums to 1.

        Args:
            o (tensor): Normalized point source position. Shape of [N, 3], x, y in range [-1, 1], z in range [-Inf, 0].
            kernel_size (int, optional): Output kernel size. Defaults to 7.
            spp (int, optional): Sample per pixel. Defaults to 1024.
            center (bool, optional): Use spot center as PSF center.

        Returns:
            psf: Shape of [N, 3, ks, ks].
        """
        psf = []
        for wvln in WAVE_RGB:
            psf.append(self.psf_diff(o, wvln=wvln, kernel_size=kernel_size, spp=spp, center=center))

        psf = torch.stack(psf, dim=0)
        return psf
    

    def psf2mtf(self, psf, diag=False):
        """ Convert 2D PSF kernel to MTF curve by FFT.

        Args:
            psf (tensor): 2D PSF tensor.

        Returns:
            freq (ndarray): Frequency axis.
            tangential_mtf (ndarray): Tangential MTF.
            sagittal_mtf (ndarray): Sagittal MTF.
        """
        psf = psf.cpu().numpy()
        x = np.linspace(-1, 1, psf.shape[1]) * self.pixel_size * psf.shape[1] / 2
        y = np.linspace(-1, 1, psf.shape[0]) * self.pixel_size * psf.shape[0] / 2

        if diag:
            raise Exception('Diagonal PSF is not tested.')
        else:
            # Extract 1D PSFs along the sagittal and tangential directions
            center_x = psf.shape[1] // 2
            center_y = psf.shape[0] // 2
            sagittal_psf = psf[center_y, :]
            tangential_psf = psf[:, center_x]

            # Fourier Transform to get the MTFs
            sagittal_mtf = np.abs(np.fft.fft(sagittal_psf))
            tangential_mtf = np.abs(np.fft.fft(tangential_psf))

            # Normalize the MTFs
            sagittal_mtf /= sagittal_mtf.max()
            tangential_mtf /= tangential_mtf.max()

            delta_x = self.pixel_size

            # Create frequency axis in cycles/mm
            freq = np.fft.fftfreq(psf.shape[0], delta_x)

            # Only keep the positive frequencies
            positive_freq_idx = freq > 0

            return freq[positive_freq_idx], tangential_mtf[positive_freq_idx], sagittal_mtf[positive_freq_idx]



    
    # ====================================================================================
    # Geometrical optics 
    #   1. Focus-related functions
    #   2. FoV-related functions
    #   3. Pupil-related functions
    # ====================================================================================

    # ---------------------------
    # 1. Focus-related functions
    # ---------------------------
    @torch.no_grad()
    def calc_bfl(self):
        """ Compute back focal length (BFL). 

            BFL: Distance from the second principal point to sensor plane.
        """
        return self.d_sensor - self.calc_principal()[1]


    @torch.no_grad()
    def calc_efl(self):
        """ Compute effective focal length (EFL). Effctive focal length is also commonly used to compute F/#.

            EFL: Defined by FoV and sensor radius.
        """
        efl = self.r_last / np.tan(self.hfov)
        return efl


    @torch.no_grad()
    def calc_eqfl(self):
        """ 35mm equivalent focal length.

            35mm sensor: 36mm * 24mm
        """
        eqfl = 21.63 / np.tan(self.hfov)
        return eqfl


    @torch.no_grad()
    def calc_foc_dist(self, wvln=DEFAULT_WAVE, spp=4096):
        """ Compute the focus distance.

            Rays start from sensor center and trace to the object space, the focus distance is negative.
        """
        # => Sample point source rays from sensor center
        o1 = torch.Tensor([0, 0, self.d_sensor]).repeat(spp, 1)
        o2 = self.surfaces[0].surface_sample(spp)   # A simple method is to sample from the first surface.
        d = o2 - o1
        ray = Ray(o1, d, normalized=False, wavelength=wvln)

        # => Trace rays to the object space and compute focus distance
        ray, _, _ = self.trace(ray)
        t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2) # The solution for the nearest distance.
        focus_p = ((ray.o[...,2] - ray.d[...,2] * t) * ray.ra).cpu().numpy()
        focus_p = focus_p[~np.isnan(focus_p) & (focus_p < 0)]
        focus_dist = float(statistics.median(focus_p))

        return focus_dist


    @torch.no_grad()
    def refocus_inf(self):
        """ Shift sensor to get the best center focusing.
        """
        # compute new sensor position
        M = 100
        d_sensor_new = []
        for i in range(3):
            ray = self.sample_parallel_2D(R=self.surfaces[0].r, M=M, wavelength=WAVE_RGB[i])
            ray, _, _ = self.trace(ray)
            t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2)
            focus_p = (ray.o[...,2] - ray.d[...,2] * t).cpu().numpy()
            focus_p = focus_p[~np.isnan(focus_p)]
            d_sensor_new.append(stats.trim_mean(focus_p, 0.1))
        
        d_sensor_new = (d_sensor_new[0] + d_sensor_new[2]) / 2
        if d_sensor_new < 0:
            print('get a concave lens.')

        # update sensor
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()


    @torch.no_grad()
    def refocus(self, depth=DEPTH):
        """ Refocus the lens to a given depth by changing sensor position.
        """
        # Consider RGB three wavelengths
        d_sensor_new = []
        for i in range(3):
            # Sample point source rays and trace to image space
            o = self.surfaces[0].surface_sample(GEO_SPP)
            d = o - torch.tensor([0, 0, depth])
            ray = Ray(o, d, normalized=False, wavelength=WAVE_RGB[i], device=self.device)
            ray, _, _ = self.trace(ray)

            # Use least-squares solution as the in-focus position
            t = (ray.d[...,0]*ray.o[...,0] + ray.d[...,1]*ray.o[...,1]) / (ray.d[...,0]**2 + ray.d[...,1]**2)
            t = t * ray.ra
            focus_d = (ray.o[...,2] - ray.d[...,2] * t).cpu().numpy()
            focus_d = focus_d[~np.isnan(focus_d)]
            focus_d = focus_d[focus_d>0]
            d_sensor_new.append(statistics.median(focus_d))
        
        d_sensor_new = np.mean(d_sensor_new)
        if d_sensor_new < 0:
            print('get a concave lens.')

        # Update sensor position
        self.d_sensor = d_sensor_new

        # FoV will be slightly changed
        self.post_computation()


    # ---------------------------
    # 2. FoV-related functions
    # ---------------------------
    @torch.no_grad()
    def calc_fov(self):
        """ Compute half diagonal fov.

            Shot rays from edge of sensor, trace them to the object space and compute
            angel, output rays should be parallel and the angle is half of fov.
        """
        # sample rays going out from edge of sensor, shape [M, 3] 
        M = 100
        pupilz, pupilx = self.exit_pupil()
        o1 = torch.zeros([M, 3])
        o1[:,0] = self.r_last if torch.is_tensor(self.r_last) else torch.Tensor([self.r_last])
        o1[:,2] = self.d_sensor if torch.is_tensor(self.d_sensor) else torch.Tensor([self.d_sensor])

        x2 = torch.linspace(-pupilx, pupilx, M)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2-o1, normalized=False, device=self.device)
        ray, _, _ = self.trace(ray)

        # compute fov
        tan_fov = ray.d[...,0]/ray.d[...,2]
        fov = torch.atan(torch.sum(tan_fov * ray.ra) / torch.sum(ray.ra))

        if torch.isnan(fov):
            print('computed fov is NaN, use 0.5 rad instead.')
            fov = 0.5
        else:
            fov = fov.item()
        
        return fov


    @torch.no_grad()
    def calc_magnification3(self, depth):
        """ Use mapping relationship to compute magnification. The computed magnification is very accurate.

            Advatages: can use many data points to reduce error.
            Disadvantages: due to distortion, some data points contain error
        """
        M = 21
        spp = 512
        ray = self.sample_point_source(M=M, spp=spp, depth=depth, R=-depth*np.tan(self.hfov)*0.5, pupil=True)
        
        # map r1 from object space to sensor space, ground-truth
        o1 = ray.o.detach()[..., 0:2]
        o1 = torch.flip(o1, [1,2])
        
        ray, _, _ = self.trace(ray, ignore_aper=False)
        o2 = ray.project_to(self.d_sensor)

        # use 1/4 part of regions to compute magnification, also to avoid zero values on the axis
        x1 = o1[0,:,:,0]
        y1 = o1[0,:,:,1]
        x2 = torch.sum(o2[...,0] * ray.ra, axis=0)/ torch.sum(ray.ra, axis=0).add(EPSILON)
        y2 = torch.sum(o2[...,1] * ray.ra, axis=0)/ torch.sum(ray.ra, axis=0).add(EPSILON)

        mag_x = x1 / x2
        tmp = mag_x[:M//2,:M//2]
        mag = 1 / torch.mean(tmp[~tmp.isnan()]).item()

        if mag == 0:
            scale = - depth * np.tan(self.hfov) / self.r_last
            return 1 / scale

        return mag


    @torch.no_grad()
    def calc_principal(self):
        """ Compute principal planes.
            Shot parallel rays from object space and trace it to the sensor, extend 
            incident rays, correpsponding output rays to get intersection points. 
            The projection on the z axis should be the principal. 
        """
        M = 20

        # backward ray tracing, compute the first principal
        ray = self.sample_parallel_2D(R=self.surfaces[0].r, M=M, forward=False)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[...,0] - inc_ray.o[...,0]) / out_ray.d[...,0]
        z = inc_ray.o[...,2] + inc_ray.d[...,2] * t
        principal1 = np.nanmean(z.cpu().numpy()) 

        # forward ray tracing, compute the second principal
        ray = self.sample_parallel_2D(R=self.surfaces[0].r, M=M, forward=True)
        inc_ray = ray.clone()
        out_ray, _, _ = self.trace(ray)

        t = (out_ray.o[...,0] - inc_ray.o[...,0]) / out_ray.d[...,0]
        z = inc_ray.o[...,2] + inc_ray.d[...,2] * t
        principal2 = np.nanmean(z.cpu().numpy()) 

        return principal1, principal2


    @torch.no_grad()
    def calc_scale_pinhole(self, depth):
        """ Assume the first principle point is at (0, 0, 0), use pinhole camera to calculate the scale factor.
        """
        scale = - depth * np.tan(self.hfov) / self.r_last
        # return scale.item()
        return scale
    

    @torch.no_grad()
    def calc_scale_ray(self, depth):
        """ Use ray tracing to compute scale factor.
        """
        scale = 1 / self.calc_magnification3(depth)
        return scale


    @torch.no_grad()
    def chief_ray(self):
        """ Compute chief ray. We can use chief ray for fov, magnification.
            Chief ray, a ray goes through center of aperture.
        """
        # sample rays with shape [M, 3]
        M = 1000
        pupilz, pupilx = self.exit_pupil()
        o1 = torch.zeros([M, 3])
        o1[:,0] = pupilx
        o1[:,2] = self.d_sensor
        
        x2 = torch.linspace(-pupilx, pupilx, M)
        y2 = torch.full_like(x2, 0)
        z2 = torch.full_like(x2, pupilz)
        o2 = torch.stack((x2, y2, z2), axis=-1)

        ray = Ray(o1, o2-o1, normalized=False, device=self.device)
        inc_ray = ray.clone()
        ray, _, _ = self.trace(ray, lens_range=list(range(self.aper_idx, len(self.surfaces))))

        center_x = torch.min(torch.abs(ray.o[:,0]))
        center_idx = torch.where(torch.abs(ray.o[:,0])==center_x)
        
        return inc_ray.o[center_idx,:], inc_ray.d[center_idx,:]


    # ---------------------------
    # 3. Pupil-related functions
    # ---------------------------
    @torch.no_grad()
    def exit_pupil(self, expand_factor=1.02):
        """ Sample **forward** rays to compute z coordinate and radius of exit pupil. 
            Exit pupil: ray comes from sensor to object space. 
        """
        return self.entrance_pupil(entrance=False, expand_factor=expand_factor)


    @torch.no_grad()
    def entrance_pupil(self, M=32, entrance=True, compute=True, expand_factor=1.02):
        """ We sample **backward** rays, return z coordinate and radius of entrance pupil. 
            Entrance pupil: how many rays can come from object space to sensor. 

            When we only consider rays from very far points.

            M should not be too large.
        """
        if self.aper_idx is None:
            if entrance:
                return self.surfaces[0].d.item(), self.surfaces[0].r
            else:
                return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # sample M forward rays from edge of aperture to last surface.
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r
        ray_o = torch.tensor([[aper_r, 0, aper_z]]).repeat(M, 1)

        phi = torch.arange(-0.5, 0.5, 1.0/M)
        if entrance:
            d = torch.stack((
                torch.sin(phi),
                torch.zeros_like(phi),
                -torch.cos(phi)
            ), axis=-1)
        else:
            d = torch.stack((
                torch.sin(phi),
                torch.zeros_like(phi),
                torch.cos(phi)
            ), axis=-1)

        ray = Ray(ray_o, d, normalized=False)
        ray.to(self.device)

        # ray tracing
        if entrance:
            lens_range = range(0, self.aper_idx)
            ray,_,_ = self.trace(ray, lens_range=lens_range)
        else:
            lens_range = range(self.aper_idx+1, len(self.surfaces))
            ray,_,_ = self.trace(ray, lens_range=lens_range)

        # compute intersection. o1+d1*t1 = o2+d2*t2
        pupilx = []
        pupilz = []
        for i in range(M):
            for j in range(i+1, M):
                if ray.ra[i] !=0 and ray.ra[j]!=0:
                    d1x, d1z, d2x, d2z = ray.d[i,0], ray.d[i,2], ray.d[j,0], ray.d[j,2]
                    o1x, o1z, o2x, o2z = ray.o[i,0], ray.o[i,2], ray.o[j,0], ray.o[j,2]
                    
                    # Method 2: manually solve
                    Adet = - d1x * d2z + d2x * d1z
                    B1 = -d1z*o1x+d1x*o1z
                    B2 = -d2z*o2x+d2x*o2z
                    oz = (- B1 * d2z + B2 * d1z) / Adet
                    ox = (B2 * d1x - B1 * d2x) / Adet
                    
                    pupilx.append(ox.item())
                    pupilz.append(oz.item())
        
        if len(pupilx)==0:
            raise Exception('Abandoned method.')

        avg_pupilx = stats.trim_mean(pupilx, 0.1)
        avg_pupilz = stats.trim_mean(pupilz, 0.1)
        if np.abs(avg_pupilz) < EPSILON:
            avg_pupilz = 0
        
        return avg_pupilz, avg_pupilx
    

    
    # ====================================================================================
    # Lens operation 
    #   1. Set lens parameters
    #   2. Lens operation (init, reverse, spherize)
    #   3. Lens pruning
    # ====================================================================================

    # ---------------------------
    # 1. Set lens parameters
    # ---------------------------
    def set_aperture(self, fnum=None, foclen=None, aper_r=None):
        """ Change aperture radius.
        """
        if aper_r is None:
            if foclen is None:
                foclen = self.calc_efl()
            aper_r = foclen / fnum / 2
            self.surfaces[self.aper_idx].r = aper_r
        else:
            self.surfaces[self.aper_idx].r = aper_r
        
        self.fnum = self.foclen / aper_r / 2


    def set_param(self, param_str, param_value):
        """ Change surface parameter.
            Should move to IO part.
        """
        exec(f'self.{param_str} = param_value')

    
    def set_target_fov_fnum(self, hfov, fnum, imgh=None):
        """ Set FoV, ImgH and F number, only use this function to assign design targets.

            This function now only works for aperture in the front
        """
        if imgh is not None:
            self.r_last = imgh / 2
        self.hfov = hfov
        self.fnum = fnum
        
        self.foclen = self.calc_efl()
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].r = aper_r

    
    def set_fov(self, hfov):
        """ Manually set FoV when FoV is not calculated correctly.

        """
        self.hfov = hfov
        self.foclen = self.calc_efl()
        self.fnum = self.foclen / self.surfaces[self.aper_idx].r / 2

    # ---------------------------
    # 3. Lens pruning
    # ---------------------------
    @torch.no_grad()
    def prune(self, outer=None):
        self.pruning_v2(outer=outer)

        
    @torch.no_grad()
    def pruning_v2(self, outer=None, surface_range=None):
        """ Prune surfaces to the minimum height that allows all valid rays to go through.

        Args:
            outer (float, optional): Outer margin. Defaults to None.
            surface_range (list, optional): Surface range to prune. Defaults to None.
        """
        if outer is None:
            outer = self.r_last * 0.05
        
        if surface_range is None:
            surface_range = self.find_diff_surf()


        # ==> 1. Reset lens to maximum height(sensor radius)
        for i in surface_range:
            self.surfaces[i].r = self.r_last

        # ==> 2. Prune to reserve valid surface height
        # sample maximum fov rays to compute valid surface height
        aper = self.surfaces[0].r
        view = self.hfov if self.hfov is not None else np.arctan(self.r_last/self.d_sensor)
        ray = self.sample_parallel_2D(R=aper, view=np.rad2deg(view), M=11, entrance_pupil=False)

        ps, oss = self.trace_to_sensor(ray=ray, record=True)
        for i in surface_range:
            height = []
            for os in oss:  # iterate all rays
                try:
                    height.append(os[i+1][0])   # the second index 0 means x coordinate
                except:
                    continue

            try:
                self.surfaces[i].r = max(height) + outer
            except:
                continue
        
        # ==> 3. Front surface should be smaller than back surface. This does not apply to fisheye lens.
        for i in surface_range[:-1]:
            if self.materials[i].A < self.materials[i+1].A:
                self.surfaces[i].r = min(self.surfaces[i].r, self.surfaces[i+1].r)

        # ==> 4. Remove nan part, also the maximum height should not exceed sensor radius
        for i in surface_range:
            max_height = min(self.surfaces[i].max_height(), self.r_last)
            self.surfaces[i].r = min(self.surfaces[i].r, max_height)


    @torch.no_grad()
    def correct_overlap(self, d_aper=0.1):
        """ If surfaces overlap, move a small distance.

            This function is not commonly used. But we can use it as a less strong version, maybe after each step??
        """
        diff_surf_range = self.find_diff_surf()

        # ==> Fix aperture distance. Only for aperture at the first surface.
        aper_idx = self.aper_idx
        if aper_idx == 0:
            # If the first surface is concave, use the maximum negative sag. 
            aper_r = self.surfaces[aper_idx].r
            sag1 = - self.surfaces[aper_idx+1].surface(aper_r, 0).item()
            if sag1 > 0:
                d_aper += sag1

            # Update aperture position.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in diff_surf_range:
                self.surfaces[i].d -= delta_aper

        # ==> Correct overlap 
        for i in diff_surf_range:
            if self.surfaces[i].d < self.surfaces[i-1].d:
                self.surfaces[i].d += 0.2

        # self.post_computation()


    @torch.no_grad()
    def correct_shape(self, d_aper=0.2):
        """ Correct wrong lens shape during the training.
        """
        aper_idx = self.aper_idx
        diff_surf_range = self.find_diff_surf()
        shape_changed = False

        # ==> Rule 1: Move the first surface to z = 0
        if self.surfaces[0].d < 0:
            move_back = self.surfaces[0].d.item()
            for surf in self.surfaces:
                surf.d += move_back
            self.d_sensor += move_back


        # ==> Rule 2: Move lens group to get a fixed aperture distance. Only for aperture at the first surface.
        if aper_idx == 0:
            # If the first surface is concave, use the maximum negative sag. 
            aper_r = self.surfaces[aper_idx].r
            sag1 = - self.surfaces[aper_idx+1].surface(aper_r, 0).item()
            if sag1 > 0:
                d_aper += sag1

            # Update position of all surfaces.
            delta_aper = self.surfaces[1].d.item() - d_aper
            for i in diff_surf_range:
                self.surfaces[i].d -= delta_aper

        
        # ==> Rule 3: If two surfaces overlap (at center), seperate them by a small distance
        for i in diff_surf_range:
            if self.surfaces[i].d < self.surfaces[i-1].d:
                self.surfaces[i].d += 0.2
                shape_changed = True
                print('single surface d changed.')

        # ==> Rule 4: Prune all surfaces
        self.pruning_v2()

        return shape_changed





    # ====================================================================================
    # Visualization.
    # ====================================================================================
    
    @torch.no_grad()
    def analysis(self, save_name='./test', back_raytrace=False, render=False, multi_plot=False, draw_spot_diagram=False, plot_invalid=True, zmx_format=True, depth=DEPTH, point_depth=None, render_unwarp=False, lens_title=None):
        """ Analyze the optical system by generating a set of parallel rays
            draw setup and spot diagram. 
        """
        # ---------------------------------
        # draw light path
        # ---------------------------------
        self.plot_setup2D_with_trace(filename=save_name, multi_plot=multi_plot, entrance_pupil=True, plot_invalid=plot_invalid, zmx_format=zmx_format, lens_title=lens_title, depth=point_depth)

        # ---------------------------------
        # draw spot diagram
        # ---------------------------------
        if draw_spot_diagram:
            self.draw_spot_diagram(save_name=save_name) 

        # ---------------------------------
        # calculate RMS error
        # ---------------------------------
        rms_avg, rms_radius_on_axis, rms_radius_off_axis = self.analysis_rms()
        print(f'Avg RMS spot size (radius): {round(rms_avg.item()*1000,3)}um, On-axis RMS radius: {round(rms_radius_on_axis.item()*1000,3)}um, Off-axis RMS radius: {round(rms_radius_off_axis.item()*1000,3)}um')

        return rms_avg


    @torch.no_grad()      
    def plot_setup2D_with_trace(self, filename, views=[0], M=9, depth=None, entrance_pupil=True, zmx_format=False, plot_invalid=True, multi_plot=False, lens_title=None):
        """ Plot lens setup with rays.
        """
        # ==> Title
        if lens_title is None:
            if self.aper_idx is not None:
                lens_title = f'FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_F/{round(self.fnum,2)}_DIAG{round(self.r_last*2, 2)}mm_FocLen{round(self.foclen,2)}mm'
            else:
                lens_title = f'FoV{round(2*self.hfov*57.3, 1)}({int(self.calc_eqfl())}mm EFL)_DIAG{round(self.r_last*2, 2)}mm_FocLen{round(self.foclen,2)}mm'
        

        # ==> Plot RGB seperately
        if multi_plot:
            R = self.surfaces[0].r
            views = np.linspace(0, np.rad2deg(self.hfov)*0.99, num=7)
            colors_list = 'rgb'
            fig, axs = plt.subplots(1, 3, figsize=(24, 6))
            fig.suptitle(lens_title)

            for i, wavelength in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.plot_setup2D(ax=ax, fig=fig, show=False, zmx_format=zmx_format)

                for view in views:
                    ray = self.sample_parallel_2D(R, wavelength, view=view, M=M, entrance_pupil=entrance_pupil)
                    ps, oss = self.trace_to_sensor(ray=ray, record=True)
                    ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra)
                    ax.axis('off')

            fig.savefig(f"{filename}.png", bbox_inches='tight', format='png', dpi=300)
            plt.close()
        

        # ==> Plot RGB in one figure
        else:
            R = self.surfaces[0].r
            colors_list = 'bgr'
            views = [0, np.rad2deg(self.hfov)*0.707, np.rad2deg(self.hfov)*0.99]
            aspect = self.sensor_res[1] / self.sensor_res[0]
            ax, fig = self.plot_setup2D(show=False, zmx_format=zmx_format)
            
            for i, view in enumerate(views):
                if depth is None:
                    ray = self.sample_parallel_2D(R, WAVE_RGB[2-i], view=view, M=M, entrance_pupil=entrance_pupil)
                else:
                    ray = self.sample_point_source_2D(depth=depth, view=view, M=M, entrance_pupil=entrance_pupil, wavelength=WAVE_RGB[2-i])
                        
                ps, oss = self.trace_to_sensor(ray=ray, record=True)
                ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i], plot_invalid=plot_invalid, ra=ray.ra)

            ax.axis('off')
            ax.set_title(lens_title)
            fig.savefig(f"{filename}.png", bbox_inches='tight', format='png', dpi=600)
            plt.close()

    
    def plot_raytraces(self, oss, ax=None, fig=None, color='b-', show=True, p=None, valid_p=None, plot_invalid=True, ra=None):
        """ Plot light path. 

        Args:
            oss (_type_): list storing all intersection points.
            ax (_type_, optional): _description_. Defaults to None.
            fig (_type_, optional): _description_. Defaults to None.
            color (str, optional): _description_. Defaults to 'b-'.
            show (bool, optional): _description_. Defaults to True.
            p (_type_, optional): _description_. Defaults to None.
            valid_p (_type_, optional): _description_. Defaults to None.
            plot_invalid (bool, optional): if we want to plot invalid rays and observe them or not.
            ra (_type_, optional): tensor storing validity of rays.

        Returns:
            _type_: _description_
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D(show=False)
        else:
            show = False

        for i, os in enumerate(oss):
            o = torch.Tensor(np.array(os)).to(self.device)
            x = o[...,0]
            z = o[...,2]

            o = o.cpu().detach().numpy()
            z = o[...,2].flatten()
            x = o[...,0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i,0])
                    z = np.append(z, p[i,2])

            if plot_invalid:
                ax.plot(z, x, color, linewidth=1.0)
            elif ra[i]>0:
                ax.plot(z, x, color, linewidth=1.0)
            
        if show: 
            plt.show()
        else: 
            plt.close()

        return ax, fig


    def plot_setup2D(self, ax=None, fig=None, show=True, color='k', with_sensor=True, zmx_format=False, fix_bound=False):
        """ Draw experiment setup.
        """
        # to world coordinate
        def plot(ax, z, x, color):
            p = torch.stack((x, torch.zeros_like(x, device=self.device), z), axis=-1)
            p = p.cpu().detach().numpy()
            ax.plot(p[...,2], p[...,0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R # [mm]

            # wedge length
            z = torch.linspace(d.item() - APERTURE_WEDGE_LENGTH, d.item() + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            
            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R+APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R-APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(ax, z, x, color)
        
        
        # =============> Function starts here <============== 
        # If no ax is given, generate a new one.
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(5,5))
        else:
            show=False

        if len(self.surfaces) == 1: # if there is only one surface, then it should be aperture
            draw_aperture(ax, self.surfaces[0], color)
        else:
            if with_sensor:
                # here we use diagonal distance as image height
                self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0, device=self.device))

            # ==> Draw surface
            for i, s in enumerate(self.surfaces):
                # => find and draw aperture
                if i < len(self.surfaces)-1:
                    # Draw aperture
                    if self.materials[i].A < 1.0003 and self.materials[i+1].A < 1.0003: # both are AIR
                        draw_aperture(ax, s, color)
                        continue

                # => Draw spherical/aspherical surface
                r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device) # aperture sampling
                z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))   # graw surface
                plot(ax, z, r, color)

            
            # ==> Draw boundary, connect two surfaces
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003: # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag      = s.surface_with_offset(r, 0.0)

                    if zmx_format:
                        z = torch.stack((sag_prev, sag_prev, sag))
                        x = torch.Tensor(np.array([[r_prev], [r], [r]])).to(self.device)
                    else:
                        z = torch.stack((sag_prev, sag))
                        x = torch.Tensor(np.array([[r_prev], [r]])).to(self.device)

                    plot(ax, z, x, color)
                    plot(ax, z,-x, color)
                    s_prev = s

            # ==> Remove sensor plane
            if with_sensor:
                self.surfaces.pop()
        
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        
        if fix_bound:
            ax.set_aspect('equal')
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect('equal', adjustable='datalim', anchor='C') 
            ax.minorticks_on() 
            ax.set_xlim(-0.5, 7.5) 
            ax.set_ylim(-4, 4)
            plt.autoscale()

        return ax, fig


    @torch.no_grad()
    def draw_psf(self, grid=7, depth=DEPTH, ks=51, log_scale=False, quater=True, save_name='./psf.png'):
        """ Draw PSF grid at a certain depth. Will draw M x M PSFs, each of size ks x ks.
        """
        spp = ks**2 if ks > 64 else 4096
        points = self.point_source_grid(depth=depth, grid=grid, quater=quater)
        psfs = []
        num_h, num_w = points.shape[:2]
        for i in range(num_h):
            for j in range(num_w):
                psf = self.psf_diff_color(o=points[i, j], kernel_size=ks, center=True, spp=spp)
                psf /= psf.max()

                if log_scale:
                    psf = torch.log(psf + EPSILON)
                    psf = (psf - psf.min()) / (psf.max() - psf.min())

                psfs.append(psf)

        psf_grid = make_grid(psfs, nrow=num_w, padding=1, pad_value=0.0)
        save_image(psf_grid, save_name, normalize=True)


    @torch.no_grad()
    def draw_psf_radial(self, M=3, depth=DEPTH, ks=51, log_scale=False, save_name='./psf_radial.png'):
        """ Draw radial PSF (45 deg). Will draw M PSFs, each of size ks x ks.  
        """
        x = torch.linspace(0, 1, M)
        y = torch.linspace(0, 1, M)
        z = torch.full_like(x, depth)
        points = torch.stack((x, y, z), dim=-1)
        
        psfs = []
        for i in range(M):
            # Scale PSF for a better visualization
            psf = self.psf_diff_color(o=points[i], kernel_size=ks, center=True, spp=4096)
            psf /= psf.max()

            if log_scale:
                psf = torch.log(psf + EPSILON)
                psf = (psf - psf.min()) / (psf.max() - psf.min())
            
            psfs.append(psf)

        psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
        save_image(psf_grid, save_name, normalize=True)


    @torch.no_grad()
    def draw_spot_diagram(self, M=7, depth=DEPTH, save_name=None):
        """ Draw spot diagram of the lens.
            
            Shot rays from grid points in object space, trace to sensor and visualize.

            (x, y) coordinates are flipped so that we can directly use it for convolution.
        """
        mag = self.calc_magnification3(depth)

        # sample and trace rays
        ray = self.sample_point_source(M=M, R=self.sensor_size[0]/2/mag, depth=depth, spp=1024, pupil=True)
        ray, _, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        o2 = - ray.o.clone().cpu().numpy()
        ra = ray.ra.clone().cpu().numpy()

        # ==> plot multiple spot diagram in one figure
        fig, axs = plt.subplots(M, M, figsize=(38,38))
        for i in range(M):
            for j in range(M):
                ra_ = ra[:,i,j]
                x, y = o2[:,i,j,0], o2[:,i,j,1]
                x, y = x[ra_>0], y[ra_>0]
                xc, yc = x.sum()/ra_.sum(), y.sum()/ra_.sum()

                # scatter plot
                axs[i, j].scatter(x, y, 1, 'black')
                axs[i, j].scatter([xc], [yc], None, 'r', 'x')
                axs[i, j].set_aspect('equal', adjustable='datalim')
        
        if save_name is None:
            plt.savefig(f'./spot{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)
        else:
            plt.savefig(f'{save_name}_spot{-depth}mm.png', bbox_inches='tight', format='png', dpi=300)

        plt.close()


    @torch.no_grad()
    def draw_spot_radial(self, M=3, depth=DEPTH, save_name=None):
        """ Draw radial spot diagram of the lens.

        Args:
            M (int, optional): field number. Defaults to 3.
            depth (float, optional): depth of the point source. Defaults to DEPTH.
            save_name (string, optional): filename to save. Defaults to None.
        """
        mag = self.calc_magnification3(depth)

        # ==> Sample and trace rays
        ray = self.sample_point_source(M=M*2-1, R=self.sensor_size[0]/2/mag, depth=depth, spp=1024, pupil=True, wavelength=589.3)
        ray, _, _ = self.trace(ray)
        ray.propagate_to(self.d_sensor)
        o2 = torch.flip(ray.o.clone(), [1, 2]).cpu().numpy()
        ra = torch.flip(ray.ra.clone(), [1, 2]).cpu().numpy()

        # ==> Plot multiple spot diagram in one figure
        fig, axs = plt.subplots(1, M, figsize=(M*12,10))
        for i in range(M):
            i_bias = i + M - 1

            ra_ = ra[:,i_bias,i_bias]
            x, y = o2[:,i_bias,i_bias,0], o2[:,i_bias,i_bias,1]
            x, y = x[ra_>0], y[ra_>0]
            xc, yc = x.sum()/ra_.sum(), y.sum()/ra_.sum()

            # scatter plot
            axs[i].scatter(x, y, 12, 'black')
            axs[i].scatter([xc], [yc], 400, 'r', 'x')
            
            # better visualization
            axs[i].set_aspect('equal', adjustable='datalim')
            axs[i].tick_params(axis='both', which='major', labelsize=18)
            axs[i].spines['top'].set_linewidth(4)
            axs[i].spines['bottom'].set_linewidth(4)
            axs[i].spines['left'].set_linewidth(4)
            axs[i].spines['right'].set_linewidth(4)

        # ==> Save figure
        if save_name is None:
            plt.savefig(f'./spot{-depth}mm_radial.svg', bbox_inches='tight', format='svg', dpi=1200)
        else:
            plt.savefig(f'{save_name}_spot{-depth}mm_radial.svg', bbox_inches='tight', format='svg', dpi=1200)

        plt.close()


    @torch.no_grad()
    def draw_mtf(self, relative_fov=[0.0, 0.7, 1.0], wvlns=DEFAULT_WAVE, depth=DEPTH):
        """ Draw MTF curve of the lens. 
        """
        relative_fov = [relative_fov] if isinstance(relative_fov, float) else relative_fov
        wvlns = [wvlns] if isinstance(wvlns, float) else wvlns
        color_list = 'rgb'

        plt.figure(figsize=(6,6))
        for wvln_idx, wvln in enumerate(wvlns):
            for fov_idx, fov in enumerate(relative_fov):
                point = torch.Tensor([fov, fov, depth])
                psf = self.psf_diff(point, wvln=wvln, kernel_size=256)
                freq, mtf_tan, mtf_sag = self.psf2mtf(psf)

                fov_deg = round(fov * self.hfov * 57.3, 1)
                plt.plot(freq, mtf_tan, color_list[fov_idx], label=f'{fov_deg}(deg)-Tangential')
                plt.plot(freq, mtf_sag, color_list[fov_idx], label=f'{fov_deg}(deg)-Sagittal', linestyle='--')

        plt.legend()
        plt.xlabel('Spatial Frequency [cycles/mm]')
        plt.ylabel('MTF')

        # Save figure
        plt.savefig(f'./mtf.png', bbox_inches='tight', format='png', dpi=300)
        plt.close()

        return


    # ====================================================================================
    # Loss function
    # ====================================================================================
    def loss_infocus(self, M=31):
        """ Sample parallel rays and compute RMS loss on the sensor plane, minimize focus loss.
        """
        focz = self.d_sensor
        loss = []
        for wv in [WAVE_RGB[0], WAVE_RGB[2]]:
            ray = self.sample_parallel(fov=0.0, M=M, wavelength=wv, entrance_pupil=True)
            ray, _, _ = self.trace(ray)
            p = ray.project_to(focz)

            # calculate RMS. center point is (0, 0) and all the rays should be valid
            loss.append(torch.sum(p.abs() * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra))   # L1 loss

        L = loss[0].detach() / loss[1].detach() * loss[0] + loss[1].detach() / loss[0].detach() * loss[1]
        return L


    def analysis_rms(self, depth=DEPTH, ref=True):
        """ Compute RMS-based error. Contain both RMS errors and RMS radius.

            Reference: green ray center. In ZEMAX, chief ray is used as reference, so our result is slightly different from ZEMAX.
        """
        H = 31
        scale = self.calc_scale_ray(depth)

        # ==> Use green light for reference
        if ref:
            ray = self.sample_point_source(M=H, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True, wavelength=DEFAULT_WAVE)
            ray, _, _ = self.trace(ray, ignore_aper=False)
            p_green = ray.project_to(self.d_sensor)
            p_center_ref = (p_green * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(0.0001).unsqueeze(-1)
    
        # ==> Calculate RMS errors
        rms = []
        rms_on_axis = []
        rms_off_axis = []
        for wavelength in WAVE_RGB:
            ray = self.sample_point_source(M=H, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True, wavelength=wavelength)
            ray, _, _ = self.trace(ray, ignore_aper=False)
            o2 = ray.project_to(self.d_sensor)
            o2_center = (o2*ray.ra.unsqueeze(-1)).sum(0)/ray.ra.sum(0).add(0.0001).unsqueeze(-1)
            
            if ref:
                o2_norm = (o2 - p_center_ref) * ray.ra.unsqueeze(-1)
            else:
                o2_norm = (o2 - o2_center) * ray.ra.unsqueeze(-1)   # normalized to center (0, 0)

            rms.append(torch.sum(o2_norm**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra))
            rms_on_axis.append(torch.sum(o2_norm[:, H//2+1, H//2+1, :]**2 * ray.ra[:, H//2+1, H//2+1].unsqueeze(-1)) / torch.sum(ray.ra[:, H//2, H//2]))
            rms_off_axis.append(torch.sum(o2_norm[:, 0, 0, :]**2 * ray.ra[:, 0, 0].unsqueeze(-1)) / torch.sum(ray.ra[:, 0, 0]))

        rms_radius = (rms[0] / 3 + rms[1] / 3 + rms[2] / 3).sqrt()
        rms_radius_on_axis = max(rms_on_axis[0], rms_on_axis[1], rms_on_axis[2]).sqrt()
        rms_radius_off_axis = max(rms_off_axis[0], rms_off_axis[1], rms_off_axis[2]).sqrt()
        return rms_radius, rms_radius_on_axis, rms_radius_off_axis
    

    def loss_rms(self, depth=DEPTH, show=False):
        """ Compute RGB RMS errors, forward rms error.

            Can also revise this function to plot PSF.
        """
        # H, W = self.sensor_res
        H = 31

        # ==> PSF and RMS by patch
        scale = - depth * np.tan(self.hfov) / self.r_last

        rms = 0
        for wavelength in WAVE_RGB:
            ray = self.sample_point_source(M=H, spp=GEO_SPP, depth=depth, R=self.sensor_size[0]/2*scale, pupil=True, wavelength=wavelength)
            ray, _, _ = self.trace(ray, ignore_aper=False)
            o2 = ray.project_to(self.d_sensor)
            o2_center = (o2*ray.ra.unsqueeze(-1)).sum(0)/ray.ra.sum(0).add(0.0001).unsqueeze(-1)    
            o2_norm = (o2 - o2_center) * ray.ra.unsqueeze(-1)   # normalized to center (0, 0)

            rms += torch.sum(o2_norm**2 * ray.ra.unsqueeze(-1)) / torch.sum(ray.ra)

        return rms / 3


    def loss_surface(self, grad_bound=0.5):
        """ Surface should be smooth, aggressive shape change should be pealized. 
        """
        loss = 0
        for i in self.find_diff_surf():
            r = self.surfaces[i].r
            loss += max(self.surfaces[i]._dgd(r**2).abs(), grad_bound)

        return loss


    def loss_self_intersec(self, dist_bound=0.1, thickness_bound=0.4):
        """ Loss function designed to avoid self-intersection.

            Select some height values, then compute the distance between two surfaces.
        """
        loss = 0
        for i in self.find_diff_surf()[:-1]:
            r = torch.linspace(0.6, 1, 11).to(self.device) * self.surfaces[i].r
            z_front = self.surfaces[i].surface(r, 0) + self.surfaces[i].d
            z_next = self.surfaces[i+1].surface(r, 0) + self.surfaces[i+1].d
            dist_min = torch.min(z_next - z_front)

            if self.materials[i].name == 'air':
                loss += min(thickness_bound, dist_min)
            else:
                loss += min(dist_bound, dist_min)

        return - loss


    def loss_last_surf(self, dist_bound=0.6):
        """ The last surface should not hit the sensor plane.

            There should also be space for IR filter.
        """
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.6, 1, 11).to(self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface(r, 0) - last_surf.d
        loss = min(dist_bound, torch.min(z_last_surf))
        return - loss

    
    def loss_ray_angle(self, target=0.7, depth=DEPTH):
        """ Loss function designed to penalize large incident angle rays.

            Reference value: > 0.7
        """
        # Sample rays [spp, M, M]
        M = 11
        spp = 32
        scale = self.calc_scale_pinhole(depth)
        ray = self.sample_point_source(M=M, spp=spp, depth=DEPTH, R=scale*self.sensor_size[0]/2, pupil=True, shrink=False)

        # Ray tracing
        ray, _, _ = self.trace(ray, ignore_aper=False)

        # Loss (we want to maximize loss)
        loss = torch.sum(ray.obliq * ray.ra) / (torch.sum(ray.ra) + EPSILON)
        loss = torch.min(loss, torch.Tensor([target]).to(self.device))

        return - loss


    def loss_reg(self, w1=0.2, w2=1, w3=1):
        """ An empirical regularization loss for lens design.
        """
        loss_reg = w1 * self.loss_infocus() + w2 * self.loss_ray_angle() + w3 * (self.loss_self_intersec() + self.loss_last_surf()) + self.loss_surface()
        return loss_reg
    


    # ====================================================================================
    # Optimization
    #   1. Activate/deactivate gradients.
    #   2. Generate optimizer.
    # ====================================================================================
    
    # ---------------------------
    # 1. Gradients-related functions
    # ---------------------------
    def activate_grad(self, diff_names):
        """ Active gradients for given parameters.
        """
        for name in diff_names:
            if type(name) is str: 
                try:
                    exec('self.{}.requires_grad = True'.format(name))
                except:
                    exec('self.{name} = self.{name}.detach()'.format(name=name))
                    exec('self.{}.requires_grad = True'.format(name))



    def deactivate_grad(self, diff_names):
        """ Deactive gradients for given parameters.
        """
        for name in diff_names:
            if type(name) is str: 
                try:
                    exec('self.{}.requires_grad = False'.format(name))
                except:
                    exec('self.{name} = self.{name}.detach()'.format(name=name))
                    exec('self.{}.requires_grad = False'.format(name))



    def activate_surf(self, activate=True, diff_surf_range=None):
        """ Activate gradient for each surface.
        """
        if diff_surf_range is None:
            diff_surf_range = range(len(self.surfaces))
            if self.aper_idx is not None:
                del diff_surf_range[self.aper_idx]

        for i in diff_surf_range:
            self.surfaces[i].activate_grad(activate)


    # ---------------------------
    # 2. Optimizer-related functions
    # --------------------------- 
    def get_optimizer(self, lrs=[1e-4, 1e-4, 0, 1e-4], iterations=100, decay=0.2):
        """ Get optimizers and schedulers for different lens parameters.

        Args:
            lrs (_type_): _description_
            epochs (int, optional): _description_. Defaults to 100.
            ai_decay (float, optional): _description_. Defaults to 0.2.
        """
        diff_surf_range = self.find_diff_surf()
        self.activate_surf(activate=True, diff_surf_range=diff_surf_range)
        if len(lrs) < 4:
            for _ in range(4 - len(lrs)):
                lrs.append(0)

        # Find surface indices with specific parameters
        c_ls = []
        d_ls = []
        k_ls = []
        ai2_ls = []
        ai4_ls = []
        ai6_ls = []
        ai8_ls = []
        ai10_ls = []
        ai12_ls = []
        b3_ls = []
        b5_ls = []
        b7_ls = []
        for i, s in enumerate(self.surfaces):
            if s.c != 0:
                c_ls.append(i)
                d_ls.append(i)
            if s.k != 0:
                k_ls.append(i)
            if s.ai_degree >= 4:
                if s.ai2 != 0:
                    ai2_ls.append(i)
                ai4_ls.append(i)
                ai6_ls.append(i)
                ai8_ls.append(i)
            if s.ai_degree >= 5:
                ai10_ls.append(i)
            if s.ai_degree >= 6:
                ai12_ls.append(i)
            if s.ai_degree >= 8:
                raise Exception('Aspherical degree too high.')

        # Create optimizer for all parameter
        params = []
        if c_ls and lrs[0] > 0:
            params.append({'params': [self.surfaces[surf].c for surf in c_ls], 'lr': lrs[0]})
        if d_ls and lrs[1] > 0:
            params.append({'params': [self.surfaces[surf].d for surf in c_ls], 'lr': lrs[1]})
        if k_ls and lrs[2] > 0:
            params.append({'params': [self.surfaces[surf].k for surf in k_ls], 'lr': lrs[2]})
        if lrs[3] > 0:
            if ai2_ls:
                params.append({'params': [self.surfaces[surf].ai2 for surf in ai2_ls], 'lr': lrs[3]/decay})
            if ai4_ls:
                params.append({'params': [self.surfaces[surf].ai4 for surf in ai4_ls], 'lr': lrs[3]})
            if ai6_ls:
                params.append({'params': [self.surfaces[surf].ai6 for surf in ai6_ls], 'lr': lrs[3]*decay})
            if ai8_ls:
                params.append({'params': [self.surfaces[surf].ai8 for surf in ai8_ls], 'lr': lrs[3]*decay**2})
            if ai10_ls:
                params.append({'params': [self.surfaces[surf].ai10 for surf in ai10_ls], 'lr': lrs[3]*decay**3})
            if ai12_ls:
                params.append({'params': [self.surfaces[surf].ai12 for surf in ai12_ls], 'lr': lrs[3]*decay**4})
        if len(lrs) == 5 and lrs[4] > 0:
            params.append({'params': [self.surfaces[surf].b3 for surf in b3_ls], 'lr': lrs[4]})
            params.append({'params': [self.surfaces[surf].b5 for surf in b5_ls], 'lr': lrs[4]*decay})
            params.append({'params': [self.surfaces[surf].b7 for surf in b7_ls], 'lr': lrs[4]*decay**2})

        optimizer = torch.optim.AdamW(params)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05*iterations, num_training_steps=iterations)
        return optimizer, scheduler


    def refine(self, lrs=[5e-4, 1e-4, 0.1, 1e-4], decay=0.1, iterations=2000, test_per_iter=100, depth=DEPTH, shape_control=True, centroid=False, importance_sampling=False, result_dir='./results'):
        """ Optimize a given lens by minimizing rms errors.

        Args:
            depth (float, optional): Depth of scene images. Defaults to DEPTH.
            lrs (list): Learning rate list. Lr for [c, d, k, ai]. Defaults to [1e-3, 1e-4, 0.1, 1e-4].
            decay (float, optional): Learning rate alpha decay. Defaults to 0.1.
        """
        # Preparation
        M = 31
        spp = 256
        sample_rays_per_iter = test_per_iter if not centroid else 5 * test_per_iter
        diff_surf_range = self.find_diff_surf()
        optimizer, scheduler = self.get_optimizer(lrs, iterations, decay)
        
        result_dir = result_dir + '/' + datetime.now().strftime("%m%d-%H%M%S")+ '-DesignLensRMS'
        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            set_logger(result_dir)
        logging.info(f'lr:{lrs}, decay:{decay}, iterations:{iterations}, spp:{spp}, M:{M}.')

        # Training
        pbar = tqdm(total=iterations+1, desc='Progress', postfix={'rms': 0})
        for i in range(iterations+1):
            
            # =========================================
            # Evaluate the lens
            # =========================================
            if i % test_per_iter == 0:
                # => Save lens
                with torch.no_grad():
                    if i > 0:
                        if shape_control:
                            self.correct_shape(d_aper=0.1)

                    self.write_lensfile(f'{result_dir}/iter{i}.txt', write_zmx=False)
                    self.write_lens_json(f'{result_dir}/iter{i}.json')
                    self.analysis(f'{result_dir}/iter{i}', zmx_format=True, plot_invalid=True, multi_plot=False)


            # =========================================
            # Compute centriod and sample new rays
            # =========================================
            if i % sample_rays_per_iter == 0:
                # => Use center spot of green rays
                with torch.no_grad():
                    mag = 1 / self.calc_scale_pinhole(depth)
                    ray = self.sample_point_source(M=M, R=self.sensor_size[0]/2/mag, depth=depth, spp=spp*4, pupil=True, wavelength=WAVE_RGB[1], importance_sampling=importance_sampling)
                    xy_center_ref = - ray.o[0, :, :, :2] * mag

                    ray, _, _ = self.trace(ray)
                    ray.propagate_to(self.d_sensor)
                    xy_center = (ray.o[...,:2]*ray.ra.unsqueeze(-1)).sum(0) / ray.ra.sum(0).add(EPSILON).unsqueeze(-1)
                    
                    # center_p shape [M, M, 2]
                    if centroid:
                        center_p = xy_center
                    else:
                        center_p = xy_center_ref
            
                # => Sample new rays for training
                rays_backup = []
                for wv in WAVE_RGB:
                    ray = self.sample_point_source(M=M, R=self.sensor_size[0]/2/mag, depth=depth, spp=spp, pupil=True, wavelength=wv, importance_sampling=importance_sampling)
                    rays_backup.append(ray)
                   

            # =========================================
            # Optimize lens by minimizing rms errors
            # =========================================
            loss_rms = []
            for j, wv in enumerate(WAVE_RGB):
                # => Ray tracing
                ray = rays_backup[j].clone()
                ray, _, _ = self.trace(ray)
                xy = ray.project_to(self.d_sensor)
                xy_norm = (xy[:, :M//2, :M//2] - center_p[:M//2,:M//2,:]) * ray.ra.unsqueeze(-1)[:, :M//2, :M//2, :] # use 1/4 rays to speed up training

                # => Weight mask
                weight_mask = (xy_norm.detach()**2).sum([0, -1])
                weight_mask /= weight_mask.mean()
                weight_mask[weight_mask < 0.5] *= 0.1

                # => Compute rms loss
                l_rms = torch.sum((xy_norm**2).sum(-1) * weight_mask) / (torch.sum(ray.ra) + EPSILON)
                loss_rms.append(l_rms)

            w_reg = 0.02
            L_reg = self.loss_reg()
            L_total = sum(loss_rms) + w_reg * L_reg

            # => Back-propagation
            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(rms=sum(loss_rms).item()/3)
            pbar.update(1)

        # Finish training
        pbar.close()
        self.activate_surf(activate=False, diff_surf_range=diff_surf_range)


    # ====================================================================================
    # Lesn file IO
    # ====================================================================================
    def read_lensfile(self, filename, use_roc=False):
        """ read lens info from structured .txt file.

        Args:
            filename ([string]): txt file storing lens group info.
            
            example:
            ```
                Thorlabs-LA1986
                type  distance  roc     diameter(height)    material
                O     20        0       100                 AIR           # the first line should be 'O' type, diameter should be height
                S     0         0       25.4                N-BK7
                S     3.26      -64.38  25.4                AIR
                I     122.4     0       25.4                AIR           # the last line should be 'I' type
            ```

        Raises:
            NotImplementedError: [description]

        Returns:
            surfaces (list): the list of lens object.
            materials (list): a list of string.
            r_last (float): sensor radius. diagonal distance/2
            d_last (float): overall length of oprical system.
        """
        surfaces = []   
        materials = []  # the material after i-th surface
        ds = [] # no use for now
        with open(filename) as file:
            line_no = 0
            d_total = 0.
            for line in file:
                if line_no < 2: # first two lines are comments; ignore them
                    line_no += 1 
                else:
                    ls = line.split()
                    surface_type, d, r = ls[0], float(ls[1]), float(ls[3])/2    # d: distance, r: radius
                    roc = float(ls[2])  # radius of curvature
                    materials.append(Material(ls[4]))
                    
                    d_total += d
                    ds.append(d)

                    # --------------------------
                    # process different surfaces
                    # --------------------------
                    # Aperture
                    if surface_type == 'A': 
                        if use_roc:
                            c = 1/roc if roc!=0 else 0
                        else:
                            c = roc

                        surfaces.append(Aspheric(r, d_total, c, device=self.device))

                    # Sensor
                    # Ignore it because we want to use our own sensor.
                    elif surface_type == 'I': # Sensor
                        d_total -= d
                        ds.pop()
                        materials.pop()
                        r_last = r
                        d_last = d

                    # Mixed-type of X and B
                    elif surface_type == 'M': 
                        raise NotImplementedError()

                    # Object. Ignored.
                    elif surface_type == 'O': 
                        d_total = 0.
                        ds.pop()

                    # Spheric or aspheric surface
                    elif surface_type == 'S': 
                        if use_roc:
                            c = 1/roc if roc!=0 else 0
                        else:
                            c = roc

                        # spheric surface
                        if len(ls) <= 5:
                            surfaces.append(Aspheric(r, d_total, c, device=self.device))

                        # aspheric surface
                        elif len(ls) == 6:
                            conic = float(ls[5])
                            ai = None
                            surfaces.append(Aspheric(r, d_total, c, conic, ai, device=self.device))
                        else:
                            ai = []
                            for ac in range(5, len(ls)):
                                if ac == 5:
                                    conic = float(ls[5])
                                else:
                                    ai.append(float(ls[ac]))
                            surfaces.append(Aspheric(r, d_total, c, conic, ai, device=self.device))
                    else:
                        raise Exception('surface type not implemented.')


        return surfaces, materials, r_last, d_last


    def write_lensfile(self, filename='./test.txt', str1='optimized lens file.\n', write_zmx=False):
        """ Write lens data into a txt file. 

        Args:
            filename (str): filename with .txt
            str1 (str, optional): first line for lens file, usually lens name. Defaults to 'optimized lens file.\n'.
        """ 
        f = open(filename, 'w')
        f.writelines(str1)
        f.writelines('type    distance   roc      diameter    material\n')
        f.writelines('O   0         0         0        AIR\n')
        for i in range(len(self.surfaces)):
            # Aspheric surface
            if isinstance(self.surfaces[i], Aspheric):
                if i == 0:
                    str2 = f'S {self.surfaces[i].d.item():.4f} '
                else:
                    str2 = f'S {self.surfaces[i].d.item() - self.surfaces[i-1].d.item():.3f} '
                
                str2 = str2 + f'{self.surfaces[i].c.item():.4f} '
                str2 = str2 + f'{self.surfaces[i].r*2:.2f} '
                str2 = str2 + f'{self.materials[i+1].name} '
                
                if self.surfaces[i].k is not None:
                    str2 = str2 + f'{self.surfaces[i].k.item():.2f} '
                
                if self.surfaces[i].ai_degree > 0:
                    if self.surfaces[i].ai_degree == 4:
                        str2 = str2 + f'{self.surfaces[i].ai2.item():e} {self.surfaces[i].ai4.item():e} {self.surfaces[i].ai6.item():e} {self.surfaces[i].ai8.item():e}'
                    elif self.surfaces[i].ai_degree == 5:
                        str2 = str2 + f'{self.surfaces[i].ai2.item():e} {self.surfaces[i].ai4.item():e} {self.surfaces[i].ai6.item():e} {self.surfaces[i].ai8.item():e} {self.surfaces[i].ai10.item():e}'
                    elif self.surfaces[i].ai_degree == 6:
                        str2 = str2 + f'{self.surfaces[i].ai2.item():e} {self.surfaces[i].ai4.item():e} {self.surfaces[i].ai6.item():e} {self.surfaces[i].ai8.item():e} {self.surfaces[i].ai10.item():e} {self.surfaces[i].ai12.item():e}'
                    else:
                        for j in range(1, self.surfaces[i].ai_degree+1):
                            a = eval(f'self.surfaces[i].ai{2*j}.item()')
                            str2 = str2 + f"{a:e} "
                f.writelines(str2+'\n')
            else:
                raise Exception('Not implemented.')
        
        f.writelines(f'I {self.d_sensor-self.surfaces[-1].d.item():.3f} {0.0} {self.r_last*2:.2f} {self.materials[-1].name}')
        f.close()


        if write_zmx:
            filename = filename[:-4] + '.zmx'
            self.write_zmx(filename)


    def write_lens_json(self, filename='./test.json'):
        """ Write the lens into .json file.
        """
        data = {}
        data['foclen'] = self.foclen
        data['fnum'] = self.fnum
        data['r_last'] = self.r_last
        data['d_sensor'] = self.d_sensor
        data['sensor_size'] = self.sensor_size
        data['surfaces'] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = s.surf_dict()
            
            if i < len(self.surfaces) - 1:
                surf_dict['d_next'] = self.surfaces[i+1].d.item() - self.surfaces[i].d.item()
            else:
                surf_dict['d_next'] = self.d_sensor - self.surfaces[i].d.item()

            surf_dict['mat1'] = self.materials[i].name
            surf_dict['mat2'] = self.materials[i+1].name
            
            data['surfaces'].append(surf_dict)

        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)


    def read_lens_json(self, filename='./test.json'):
        """ Read the lens from .json file.
        """
        self.surfaces = []
        self.materials = []
        with open(filename, 'r') as f:
            data = json.load(f)
            for surf_dict in data['surfaces']:
                
                if surf_dict['type'] == 'Aspheric':
                    s = Aspheric(r=surf_dict['r'], d=surf_dict['d'], c=surf_dict['c'], k=surf_dict['k'], ai=surf_dict['ai'], device=self.device)

                elif surf_dict['type'] == 'Stop':
                    s = Aspheric(r=surf_dict['r'], d=surf_dict['d'], c=surf_dict['c'], device=self.device)
                
                elif surf_dict['type'] == 'Spheric':
                    s = Aspheric(r=surf_dict['r'], d=surf_dict['d'], c=surf_dict['c'], device=self.device)
                
                else:
                    raise Exception('Surface type not implemented.')
                
                self.surfaces.append(s)
                self.materials.append(Material(surf_dict['mat1']))

        self.materials.append(Material(surf_dict['mat2']))
        self.r_last = data['r_last']
        self.d_sensor = data['d_sensor']

        # After loading lens file
        self.find_aperture()
        self.prepare_sensor(self.sensor_res)
        self.diff_surf_range = self.find_diff_surf()
        self.post_computation()


    def write_zmx(self, filename='./test.zmx'):
        """ Write the lens into .zmx file.
        """
        # Process lensgroup data
        for s in self.surfaces:
            if isinstance(s, Aspheric) and s.ai_degree == 5:
                s.ai12 = torch.Tensor([0.0])

        # Write lens file into .zmx
        f = open(filename, 'w')
        
        # => Head file
        head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {self.surfaces[0].r*2}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 {0.707*self.hfov*57.3} {0.99*self.hfov*57.3}
WAVL 0.4861327 0.5875618 0.6562725
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 2
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
    TYPE STANDARD
    FIMP 
    CURV 0.0 0 0 0 0 ""
    HIDE 1 0 0 0 0 0 0 0 0 0
    DISZ INFINITY
"""
        f.writelines(head_str)
        
        # => Surface file
        for i, s in enumerate(self.surfaces):
            # Aspheric surface
            if isinstance(s, Aspheric):
                # Stop
                if i == 0:
                    surf_str = f"""SURF {i+1}
    STOP
    TYPE STANDARD
    FIMP 
    CURV 0.0 0 0 0 0 ""
    HIDE 1 0 0 0 0 0 0 0 0 0
    DISZ {self.surfaces[i+1].d.item()-self.surfaces[i].d.item():.6e}
    DIAM {s.r}
"""                 
                    f.writelines(surf_str)

                # Odd surface
                elif i%2 ==1 :
                    surf_str = f"""SURF {i+1}
    TYPE EVENASPH
    FIMP 
    CURV {s.c.item():.4e} 1 0 0 0 ""
    HIDE 0 0 0 0 0 0 0 0 0 0
    MIRR 2 0
    SLAB 6
    PARM 1 {s.ai2.item():.8e}
    VPAR 1
    PARM 2 {s.ai4.item():.8e}
    VPAR 2
    PARM 3 {s.ai6.item():.8e}
    VPAR 3
    PARM 4 {s.ai8.item():.8e}
    VPAR 4
    PARM 5 {s.ai10.item():.10e}
    VPAR 5
    PARM 6 {s.ai12.item():.12e}
    VPAR 6
    PARM 7 0
    PARM 8 0
    DISZ {self.surfaces[i+1].d.item()-self.surfaces[i].d.item():.6e}
    GLAS {self.materials[i+1].glassname} 0 0 {self.materials[i+1].n} {self.materials[i+1].V}
    CONI {s.k.item():.6e}
    VDSZ 0 0
    DIAM {s.r} 1 0 0 1 ""
    POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
    FLAP 0 {s.r} 0
"""
                    f.writelines(surf_str)

                # even surface
                elif i%2 ==0:
                    if i < len(self.surfaces)-1:
                        surf_str = f"""SURF {i+1}
    TYPE EVENASPH
    FIMP 
    CURV {s.c.item():.4e} 1 0 0 0 ""
    HIDE 0 0 0 0 0 0 0 0 0 0
    MIRR 2 0
    SLAB 6
    PARM 1 {s.ai2.item():.8e}
    VPAR 1
    PARM 2 {s.ai4.item():.8e}
    VPAR 2
    PARM 3 {s.ai6.item():.8e}
    VPAR 3
    PARM 4 {s.ai8.item():.8e}
    VPAR 4
    PARM 5 {s.ai10.item():.10e}
    VPAR 5
    PARM 6 {s.ai12.item():.12e}
    VPAR 6
    PARM 7 0
    PARM 8 0
    DISZ {self.surfaces[i+1].d.item()-self.surfaces[i].d.item():.6e}
    CONI {s.k.item():.6e}
    VDSZ 0 0
    DIAM {s.r} 1 0 0 1 ""
    POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
    FLAP 0 {s.r} 0
"""
                    # last even surface, distance is computed by r_last
                    else:
                        surf_str = f"""SURF {i+1}
    TYPE EVENASPH
    FIMP 
    CURV {s.c.item():.4e} 0 0 0 0 ""
    HIDE 0 0 0 0 0 0 0 0 0 0
    MIRR 2 0
    SLAB 6
    PARM 1 {s.ai2.item():.8e}
    PARM 2 {s.ai4.item():.8e}
    PARM 3 {s.ai6.item():.8e}
    PARM 4 {s.ai8.item():.8e}
    PARM 5 {s.ai10.item():.10e}
    PARM 6 {s.ai12.item():.12e}
    PARM 7 0
    PARM 8 0
    DISZ {self.d_sensor-self.surfaces[i].d.item():.6e}
    CONI {s.k.item():.6e}
    VDSZ 0 0
    DIAM {s.r} 1 0 0 1 ""
    POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
"""
                    f.writelines(surf_str)

        # => Sensor file
        sensor_str = f"""SURF {i+2}
    TYPE STANDARD
    CURV 0.
    DISZ 0.0
    DIAM {self.r_last}
"""
        f.writelines(sensor_str)
        
        f.close()



# ====================================================================================
# Other functions.
# ====================================================================================
def create_lens(rff=1.0, flange=1.0, d_aper=0.5, d_sensor=None, hfov=0.6, imgh=6, fnum=2.8, surfnum=4, dir='.'):
    """ Create a flat starting point for cellphone lens design.

        Aperture is placed 0.2mm i front of the first surface.

    Args:
        r: relative flatness factor. r = L/img_r(diag)
        flange: distance from last surface to sensor
        d_sensor: total distance of then whole system. usually as a constraint in designing cellphone lens.
        fov: half diagonal fov in radian.
        foclen: focus length.
        fnum: maximum f number.
        surfnum: number of pieces to use.
    """ 
    foclen = imgh / 2 / np.tan(hfov)
    aper = foclen / fnum / 2
    if d_sensor is None:
        d_sensor = imgh * rff # total thickness

    d_opt = d_sensor - flange - d_aper
    partition = np.clip(np.random.randn(2*surfnum-1) + 1, 1, 2) # consider aperture, position normalization
    partition = [p if i%2==0 else (0.6+0.05*i)*p for i,p in enumerate(partition)]    # distance between lenses should be small
    partition[-2] *= 1.2  # the last lens should be far from the last second one
    partition = partition/np.sum(partition)
    d_ls = partition * d_opt
    d_ls = np.insert(d_ls, 0, 0)
    
    d_total = 0
    
    mat_table = MATERIAL_TABLE
    mat_names = ['coc', 'okp4', 'pmma', 'pc', 'ps']
    
    surfaces = []
    materials = []
    
    # add aperture in the front
    surfaces.append(Aspheric(aper, 0))
    materials.append(Material(name='occluder'))
    d_total += d_aper
    
    # add lens surface
    for i in range(2*surfnum):
        # surface
        d_total += d_ls[i]
        surfaces.append(Aspheric(imgh/2, d_total)) 
        
        # material (last meterial)
        n = 1
        if i % 2 != 0:
            while(n==1):    # randomly select a new glass until n!=1
                mat_name = random.choice(mat_names)
                n, v = mat_table[mat_name]

        mat = Material(name='air') if n==1 else Material(name=mat_name) 
        materials.append(mat)
    
    # add sensor
    materials.append(Material(name='air'))
    lens = Lensgroup()
    lens.load_external(surfaces, materials, imgh/2, d_sensor)

    lens.write_lensfile(f'{dir}/starting_point_hfov{hfov}_imgh{imgh}_fnum{fnum}.txt')
    return lens


def read_zmx(filename='./test'):
        """ Read .zmx file into lens file in our format.

        Args:
            filename (str, optional): _description_. Defaults to './test'.
        """
        if filename[-4:] == '.zmx':
            filename = filename
        else:
            filename = filename + '.zmx'

        with open(filename, 'r') as f:
            content = f.read()
            # content = content.decode("utf-16")

        write_surf = False
        total_d = 0
        surfaces = []
        materials = []
        surf_mate = 'AIR'
        mate = Material(name=surf_mate)
        materials.append(mate)
        surf_k = 0.0
        
        lines = content.split('\n')
        for line in lines:
            words = list(filter(None, line.split(' '))) # remove ' '
            if len(words) == 0:
                continue
            
            if words[0] == 'SURF':
                surf_info = True
                surf_idx = int(words[1])

                # append last surface
                if surf_idx == 0 or surf_idx == 1:
                    continue
                else:
                    if surf_type == 'STANDARD':
                        surf = Aspheric(r=surf_r, d=total_d-surf_d)
                        surfaces.append(surf)
                        mate = Material(name=surf_mate)
                        materials.append(mate)
                    elif surf_type == 'EVENASPH':
                        surf_ai = [surf_a2, surf_a4, surf_a6, surf_a8, surf_a10, surf_a12, surf_a14, surf_a16]
                        surf_ai = [ai for ai in surf_ai if ai!=0]
                        surf = Aspheric(r=surf_r, d=total_d-surf_d, c=surf_c, k=surf_k, ai=surf_ai)
                        surfaces.append(surf)
                        mate = Material(name=surf_mate)
                        materials.append(mate)
                        surf_mate = 'AIR'
                    
                    continue

            elif words[0] == 'STOP':
                aper = True

            elif words[0] == 'TYPE':
                surf_type = words[1]
            
            elif words[0] == 'CURV':
                surf_c = float(words[1])

            elif words[0] == 'PARM':
                if words[1] == '1':
                    surf_a2 = float(words[2])
                elif words[1] == '2':
                    surf_a4 = float(words[2])
                elif words[1] == '3':
                    surf_a6 = float(words[2])
                elif words[1] == '4':
                    surf_a8 = float(words[2])
                elif words[1] == '5':
                    surf_a10 = float(words[2])
                elif words[1] == '6':
                    surf_a12 = float(words[2])
                elif words[1] == '7':
                    surf_a14 = float(words[2])
                elif words[1] == '8':
                    surf_a16 = float(words[2])

            elif words[0] == 'DISZ':
                surf_d = float(words[1])
                if surf_d != float('inf'):
                    total_d += surf_d

            elif words[0] == 'CONI':
                surf_k = float(words[1])

            elif words[0] == 'DIAM':
                surf_r = float(words[1])

            elif words[0] == 'GLAS':
                surf_mate = words[1]

        # Sensor
        lens = Lensgroup()
        lens.load_external(surfaces, materials, surf_r, total_d)
        lens.write_lensfile(f'{filename[:-4]}.txt')

        return lens


# ------------------------------------------------------------------------------------
# Debugging.
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
    exit()
