import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from skimage import measure
import trimesh
from pydicom.errors import InvalidDicomError
import cv2
import imageio
from func import compress, contrast, apply_gradient
import tranche
from vispy import scene


############################################################################

def compute_edges(arr):
    arr = np.asarray(arr)
    edges = np.zeros(len(arr) + 1)
    edges[1:-1] = (arr[:-1] + arr[1:]) / 2
    # Extrapolate first and last edge
    edges[0] = arr[0] - (arr[1] - arr[0]) / 2
    edges[-1] = arr[-1] + (arr[-1] - arr[-2]) / 2
    return edges

def show_slice(ax, img, index, axis, x=None, y=None, z=None):
    ax.clear()
    vmin, vmax = 0, img.max()
    if axis == 0:
        # img[index, :, :] is a (len(y), len(z)) array
        extent = [y[0], y[-1], z[-1], z[0]] if (y is not None and z is not None) else None
        aspect = abs((extent[1]-extent[0]) / (extent[3]-extent[2])) if extent is not None else 'auto'
        # print("aspect : ",aspect)
        ax.imshow(img[index, :, :], cmap='gray', extent=extent, aspect=aspect, vmin=vmin, vmax=vmax)
        ax.set_title(f'Normal 0 (Slice {index})')
        ax.set_xlabel('y' if y is not None else '')
        ax.set_ylabel('z' if z is not None else '')
    elif axis == 1:
        # img[:, index, :] is a (len(x), len(z)) array
        extent = [z[0], z[-1], x[-1], x[0]] if (x is not None and z is not None) else None
        aspect = abs((extent[1]-extent[0]) / (extent[3]-extent[2])) if extent is not None else 'auto'
        ax.imshow(img[:, index, :], cmap='gray', extent=extent, aspect=aspect, vmin=vmin, vmax=vmax)
        ax.set_title(f'Normal 1 (Slice {index})')
        ax.set_xlabel('y' if y is not None else '')
        ax.set_ylabel('x' if x is not None else '')
    elif axis == 2:
        # img[:, :, index] is a (len(x), len(y)) array
        extent = [y[0], y[-1], x[-1], x[0]] if (x is not None and y is not None) else None
        aspect = abs((extent[3]-extent[2]) / (extent[1]-extent[0])) if extent is not None else 'auto'
        ax.imshow(img[:, :, index], cmap='gray', extent=extent, aspect=aspect, vmin=vmin, vmax=vmax)
        ax.set_title(f'Normal 2 (Slice {index})')
        ax.set_xlabel('x' if x is not None else '')
        ax.set_ylabel('z' if z is not None else '')
    ax.axis('auto')
    # ax.axis('auto') resets autoscaling and may un-invert the y-axis that
    # imshow set via extent. For axial slices (axis 2) we restore it so that
    # anterior (small Y_patient = x[0]) stays at the top.
    if axis == 2 and x is not None:
        ax.set_ylim(x[-1], x[0])
                  
def clean_data(list_tranches_path):
    """Returns (sorted_paths, sorted_coords, normal_axis) for valid 512x512 DICOM slices."""
    n = len(list_tranches_path)
    paths = np.empty(n, dtype=object)
    coords = np.empty(n, dtype=float)
    acq_numbers = np.empty(n, dtype=object)
    normal_axis = None
    count = 0
    for tp in list_tranches_path:
        try:
            t = tranche.Tranche(tp)
            normal_axis = t.normal_axis()
            if t.Rows == 512 and t.Columns == 512:
                paths[count] = tp
                coords[count] = float(t.ImagePositionPatient[normal_axis])
                acq_numbers[count] = getattr(t, 'AcquisitionNumber', None)
                count += 1
        except InvalidDicomError:
            continue
    if count == 0:
        raise InvalidDicomError(f'No valid DICOM files found\nLast detected axis = {normal_axis}')
    paths, coords, acq_numbers = paths[:count], coords[:count], acq_numbers[:count]
    # Keep only slices from the most common AcquisitionNumber
    unique, counts = np.unique(acq_numbers, return_counts=True)
    majority = unique[np.argmax(counts)]
    mask = acq_numbers == majority
    paths, coords = paths[mask], coords[mask]
    idx = np.argsort(coords)
    return paths[idx], coords[idx], normal_axis
    
##############################################################################

class Image3D():
    """ Class to handle a set of dicom files, compiled into a 3D image.
    normal_axis : int
    x : list of coordinates
    y : list of coordinates
    z : list of coordinates
    shape : tuple(len(x),len(y),len(z))
    img : 3D numpy array of shape (len(x),len(y),len(z))
    """

    def __init__(self,list_tranches_path):
        """Initialize a 3D image by secifying the axis along wich the scan is done and the path to the dicom files.\n
        **normal_axis** : 0 for x, 1 for y, 2 for z.\n
        **list_tranches_path** : either a list of paths to the dicom files or path to the folder"""
        # Check if initialize with list or folder path
        if isinstance(list_tranches_path, str):
            L = sorted([list_tranches_path + '/' + f for f in os.listdir(list_tranches_path)])
        else:
            L = list_tranches_path
        usefull_tranche_path, normal_coord, self.normal_axis = clean_data(L)

        # With first slice, we can get the image position and pixel spacing
        t = tranche.Tranche(usefull_tranche_path[0])
        # t_arr = pdcp.pixel_array(usefull_tranche_path[0])
        # plt.imshow(t_arr,cmap='gray')
        # plt.show()
        image_position = np.array(t.ImagePositionPatient, dtype=np.float32)
        pixel_spacing = np.array(t.PixelSpacing, dtype=np.float32)
        v1,v2 = t.vec_plan()
        spot = [0,1,2]
        spot.remove(self.normal_axis)
        a = np.array([image_position[spot[0]] + j*pixel_spacing[0]*v1.sum() for j in range(512)])
        o = np.array([image_position[spot[1]] + j*pixel_spacing[1]*v2.sum() for j in range(512)])
        # Map normal_axis → (x, y, z)  without if/elif
        coord_map = [
            (normal_coord, a, o),  # axis 0: x=slices, y=a, z=o
            (a, normal_coord, o),  # axis 1: x=a, y=slices, z=o
            (o, a, normal_coord),  # axis 2: x=o, y=a, z=slices
        ]
        self.x, self.y, self.z = coord_map[self.normal_axis]

        self.shape = (len(self.x), len(self.y), len(self.z))
        self.img = np.zeros(self.shape, dtype=float)

        # Pre-select slice indexer once (hoisted out of loop)
        idx_fns = [
            lambda i: (i, slice(None), slice(None)),
            lambda i: (slice(None), -i - 1, slice(None)),
            lambda i: (slice(None), slice(None), i),
        ]
        get_idx = idx_fns[self.normal_axis]

        for i, tp in enumerate(tqdm(usefull_tranche_path, desc=f"Reading axis {self.normal_axis}", colour='blue')):
            sl = tranche.Tranche(tp)
            pixel_arr = sl.pixel_array.astype(float)
            slope = float(getattr(sl, 'RescaleSlope', 1))
            intercept = float(getattr(sl, 'RescaleIntercept', 0))
            pixel_arr = pixel_arr * slope + intercept  # convert to Hounsfield Units
            # For axial slices (normal_axis=2): pixel_array[row,col] maps row→Y_patient (anterior→posterior)
            # and col→X_patient (right→left). Store directly so imshow gives correct radiological view.
            # For other axes the previous rotation is kept pending validation with non-axial data.
            self.img[get_idx(i)] = pixel_arr if self.normal_axis == 2 else np.rot90(pixel_arr, 1)
                

    # Other initialization methods
    def copy(self):
        out = self.empty(self.x,self.y,self.z)
        out.img = self.img
        return out

    @classmethod
    def null(cls):
        obj = cls.__new__(cls)
        return obj
    
    @classmethod 
    def empty(cls,x,y,z):
        obj = cls.__new__(cls)
        obj.x=np.array(x)
        obj.y=np.array(y)
        obj.z=np.array(z)
        obj.shape = (len(x),len(y),len(z))
        obj.img = np.zeros(obj.shape)
        return obj
    
    @classmethod
    def mean(cls,*args,**kwargs):
        """Computes an image from several using the mean value.\n
        The mesh uses the finest image's mesh along each axis."""
        # Si aucun maillage n'est spécifié, on récupère le plus fin sur chaque axe
        if 'mesh' not in kwargs:
            X = np.zeros(1)
            Y = np.zeros(1)
            Z = np.zeros(1)
            for i,obj in enumerate(args):
                # Hypothèse grossière : les Image3D sont sur une meme plage, avec un maillage régulier
                if len(obj.x) > len(X):
                    X = obj.x
                if len(obj.y) > len(Y):
                    Y = obj.y
                if len(obj.z) > len(Z):
                    Z = obj.z
        else :
            X, Y, Z = kwargs['mesh']
        
        out = Image3D.empty(X,Y,Z)
        n=len(args)
        # Pour chaque point du maillage, on moyenne toutes les cellules auxquel le point appartient
        for i,x in tqdm(range(len(X)),desc=f'Calculating Mean Image, size {out.shape}',colour='orange'):
            x = X[i]
            for j,y in enumerate(Y):
                for k,z in enumerate(Z):
                    for obj in args :
                        out.img[i,j,k] += obj.fetch(x,y,z)
        out = out/n

    def properties(self):
        m = self.img.min()
        M = self.img.max()
        contrast = M - m
        mean = self.img.mean()
        median = np.median(self.img)
        px0 =f"{ (self.img.size - np.count_nonzero(self.img))/self.img.size:.02f} %"
        return {'dim': self.img.shape,'min': m, 'max': M, 'contrast': contrast,'valeur moyenne': mean,'valeur medianne':median,'Pixels nuls':px0}
  
    def change_contrast(self, c):
        """
        Change contrast of the image by a value c in [-100, 100].
        c < 0: decrease contrast, c > 0: increase contrast.
        """
        c = np.clip(c, -100, 100)
        # Map c to a contrast factor: 0 = no change, 100 = strong, -100 = flat
        self.img = contrast(self.img,c)
    
    def compress(self,seuil,fac=0.5):
        self.img = compress(self.img,seuil,fac)


    def fetch(self,x,y,z):
        """Fetch the value at the specified coordinates (x, y, z) in the 3D image."""
        self.z = np.array(self.z)
        self.y = np.array(self.y)
        self.x = np.array(self.x)
        z_idx = np.abs(self.z - z).argmin()
        y_idx = np.abs(self.y - y).argmin()
        x_idx = np.abs(self.x - x).argmin()
        return self.img[x_idx, y_idx, z_idx]
        
    def show(self, fig=None, axs=None,edge = False):
        """Display a mid slice of the scan in each normal direction."""
        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        slice_indices = [self.shape[0] // 2, self.shape[1] // 2, self.shape[2] // 2]
        # Compute cell edges from centers for x, y, z

        # (1) Normal 0: x fixed, show (y, z)
        axs[0].imshow( self.img[slice_indices[0], :, :], cmap='gray')
        axs[0].set_title(f'Normal X (Slice {slice_indices[0]})')
        axs[0].set_xlabel('y')
        axs[0].set_ylabel('z')

        # (2) Normal 1: y fixed, show (x, z)
        axs[1].imshow( self.img[:, slice_indices[1], :], cmap='gray')
        axs[1].set_title(f'Normal Y (Slice {slice_indices[1]})')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('z')

        # (3) Normal 2: z fixed, show (x, y)
        axs[2].imshow( self.img[:, :, slice_indices[2]], cmap='gray')
        axs[2].set_title(f'Normal Z (Slice {slice_indices[2]})')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('y')

        for i in range(3):
            axs[i].axis('auto')

        plt.tight_layout()

    def projection(self,ax,grad=False)->np.ndarray:
        """projette le scan selon un axe pour donner un simuli de radio"""
        if grad :
            proj = apply_gradient(self.img,ax,0.4,1,False).sum(ax)
        else :
            proj = self.img.sum(ax)
        return proj

    def plot_hist(self,log=False,ax = None,**kwargs):
        if ax is None :
            fig, ax = plt.subplots()
        ax.hist(self.img.flatten(), bins=256, color='b', alpha=0.7,log=log)
        if 'max_freq' in kwargs:
            plt.ylim(top=kwargs['max_freq'])
        if 'range' in kwargs:
            plt.xlim(kwargs['range'])
        else : 
            plt.xlim(0, self.img.max())
        ax.set_xlabel('Pixel Value')
        plt.grid(True)
        return ax

    def seuil(self,value):
        self.img[self.img < value] = 0

    def enhance_contrast(self):
        m = self.img.min()
        M = self.img.max()
        if M > m:
            self.img = (self.img - m) / (M - m)
            self.img = np.clip(self.img, 0, 1)*1000


    def crop_index(self,x_tokeep=None,y_tokeep=None,z_tokeep=None):
        if x_tokeep is None :
            x_tokeep = [0,len(self.x)]
        if y_tokeep is None :
            y_tokeep = [0,len(self.y)]
        if z_tokeep is None :
            z_tokeep = [0,len(self.z)]
        self.img = self.img[x_tokeep[0]:x_tokeep[1],y_tokeep[0]:y_tokeep[1],z_tokeep[0]:z_tokeep[1]] 
        self.x = self.x[x_tokeep[0]:x_tokeep[1]]
        self.y = self.y[y_tokeep[0]:y_tokeep[1]]
        self.z = self.z[z_tokeep[0]:z_tokeep[1]]
        self.shape = self.img.shape

    def unique_value(self):
        self.img[self.img > 0 ] = 1

    def clean_objects(self,size=2):
        structure = np.ones((size,size,size), dtype=bool)

        # Appliquer l'ouverture morphologique
        self.img = scipy.ndimage.binary_opening(self.img, structure=structure)

    def to_step(self, filename='result.stl', threshold=0):
        """
        Export the 3D image as a .step file using marching cubes.
        Requires: scikit-image, trimesh, and the 'step' exporter from trimesh.
        Args:
            filename (str): Output .step file path.
            threshold (float): Threshold for surface extraction (default: 0.5).
        """
        # Normalize if needed
        img = self.img
        if img.max() > 1.0:
            img = img / img.max()

        # Marching cubes to extract surface
        verts, faces, normals, _ = measure.marching_cubes(img, level=threshold)

        # Convert voxel coordinates to world coordinates
        # Assume self.x, self.y, self.z are sorted arrays of coordinates
        verts_world = np.zeros_like(verts)
        verts_world[:, 0] = np.interp(verts[:, 0], np.arange(len(self.x)), self.x)
        verts_world[:, 1] = np.interp(verts[:, 1], np.arange(len(self.y)), self.y)
        verts_world[:, 2] = np.interp(verts[:, 2], np.arange(len(self.z)), self.z)

        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals, process=False)
        mesh = mesh.smoothed(filter='laplacian', iterations=10)

        # Export to STEP using trimesh's step exporter
        try:
            mesh.export(filename, file_type='stl')
        except Exception as e:
            raise RuntimeError(f"Failed to export STEP file: {e}")
        

    def rotation_video(self, outpath='scan_anim.mp4', n_frames=72, size=512, grad=False, axis='z', pov='x', fps=10):
        """Saves a video of the 3D image rotating around a chosen axis using Vispy GPU rendering.

        Three explicit steps:
          1. Orient the volume so `axis` points up (world Z = turntable up).
          2. Place the camera along `pov`.
          3. Sweep azimuth 0→360° — a clean turntable rotation around `axis`.

        Args:
            outpath:  output path — .mp4 or .gif
            n_frames: number of frames (default 72 = 5° steps for a full 360°)
            size:     pixel resolution of the longest side (the other side is scaled proportionally)
            grad:     if True, use additive (X-ray) rendering; if False, use MIP
            axis:     data axis to rotate around — 'x', 'y', or 'z' (default)
            pov:      camera starting side — 'x' (default, side view),
                      'y' (front view), or 'z' (top-down)
            fps:      frames per second (default 10)
        """
        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
        if pov not in ('x', 'y', 'z'):
            raise ValueError(f"pov must be 'x', 'y', or 'z', got {pov!r}")

        # Normalize to [0, 1] without touching self.img
        img = self.img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Voxel spacings in mm along each patient axis.
        # img layout: axis0=Y_patient (self.x), axis1=X_patient (self.y), axis2=Z_patient (self.z)
        sp0 = abs(float(self.x[1] - self.x[0])) if len(self.x) > 1 else 1.0
        sp1 = abs(float(self.y[1] - self.y[0])) if len(self.y) > 1 else 1.0
        sp2 = abs(float(self.z[1] - self.z[0])) if len(self.z) > 1 else 1.0

        # Step 1 — Reorient so the chosen rotation axis becomes vispy axis 1 (TurntableCamera up).
        if axis == 'z':
            img = img.transpose(0, 2, 1)   # (Y_pat, Z_pat, X_pat) → Z_patient is up
            spacings = (sp0, sp2, sp1)
        elif axis == 'x':
            img = img.transpose(2, 0, 1)   # (Z_pat, Y_pat, X_pat) → Y_patient is up
            spacings = (sp2, sp0, sp1)
        else:  # axis == 'y'
            spacings = (sp0, sp1, sp2)     # X_patient already at axis 1

        # Resample to isotropic voxels (downsample to coarsest spacing).
        # This makes each voxel a unit cube in world space so the canvas aspect
        # ratio alone determines the correct physical proportions — no STTransform needed.
        from scipy.ndimage import zoom as sp_zoom
        target_sp = max(spacings)
        zoom_factors = tuple(s / target_sp for s in spacings)
        if any(abs(f - 1.0) > 0.02 for f in zoom_factors):
            img = sp_zoom(img, zoom_factors, order=1, prefilter=False)

        # Crop to tight bounding box of non-zero voxels to eliminate surrounding black space
        nz_idx = np.argwhere(img > 0)
        if nz_idx.size:
            lo = nz_idx.min(axis=0)
            hi = nz_idx.max(axis=0) + 1
            img = img[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

        method = 'additive' if grad else 'mip'

        # Canvas size: axis1 (up) → height, max(axis0, axis2) → width (rotation envelope)
        n0, n1, n2 = img.shape
        w_raw, h_raw = max(n0, n2), n1
        scale = size / max(w_raw, h_raw)
        w, h = round(w_raw * scale), round(h_raw * scale)

        canvas = scene.SceneCanvas(size=(w, h), show=True, bgcolor='black')
        canvas.show(False)
        view = canvas.central_widget.add_view()

        scene.visuals.Volume(img, parent=view.scene, method=method, cmap='grays', clim=(0, 1))

        # Step 2 — Place the camera along `pov`
        _pov_map = {'x': (0, 0), 'y': (90, 0), 'z': (0, 90)}  # (azimuth, elevation)
        init_az, init_el = _pov_map[pov]
        cam = scene.cameras.TurntableCamera(fov=0, elevation=init_el, azimuth=init_az)
        view.camera = cam
        cam.set_range()
        # Override scale_factor for a tight, margin-free fit.
        # Voxels are now isotropic so nz, ny, nx are proportional to physical mm.
        cam.scale_factor = n1 if init_el == 0 else max(n0, n2)

        # Step 3 — Rotate: pure azimuth sweep around world Z (= chosen axis)
        is_gif = outpath.lower().endswith('.gif')

        if is_gif:
            frames = []
            for i in tqdm(range(n_frames), desc='Generating GIF'):
                cam.azimuth = init_az + i * (360.0 / n_frames)
                frames.append(canvas.render(alpha=False))  
            canvas.close()
            imageio.mimsave(outpath, frames, fps=fps, loop=0)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))
            for i in tqdm(range(n_frames), desc='Generating Video'):
                cam.azimuth = init_az + i * (360.0 / n_frames)
                frame = canvas.render(alpha=False)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            canvas.close()

            
            

#########################################################################################

# def pad_to_shape(arr, shape):
#     pad_width = []
#     for s, t in zip(arr.shape, shape):
#         total = max(t - s, 0)
#         before = total // 2
#         after = total - before
#         pad_width.append((before, after))
#     return np.pad(arr, pad_width, mode='constant'), pad_width

# def unpad(arr,pad_width):
#     slices = []
#     for (before, after) in pad_width:
#         start = before
#         end = -after if after > 0 else None
#         slices.append(slice(start, end))
#     return arr[tuple(slices)]


# def ressemblance(img1:Image3D,img2:Image3D):
#     X = img1.x
#     Y = img1.y
#     Z = img1.z
#     score = 0
#     for i in tqdm(range(0,len(X),2),desc='Calcul du Score'):
#         x=X[i]
#         for j in range(0,len(Y),2):
#             y = Y[j]
#             for k in range(0,len(Z),2):
#                 z = Z[k]
#                 score += img1.fetch(x,y,z)*img2.fetch(x,y,z)
#     return score

# def standardize(obj:Image3D,*args):
#     if len(args)>0:
#         xmins = [obj.x.min()] + [args[i].x.min() for i in range(len(args)) ]
#         ymins = [obj.y.min()] + [args[i].y.min() for i in range(len(args)) ]
#         zmins = [obj.z.min()] + [args[i].z.min() for i in range(len(args)) ]
#         xmaxs = [obj.x.max()] + [args[i].x.max() for i in range(len(args)) ]
#         ymaxs = [obj.y.max()] + [args[i].y.max() for i in range(len(args)) ]
#         zmaxs = [obj.z.max()] + [args[i].z.max() for i in range(len(args)) ]
#         xmin = min(xmins)
#         ymin = min(ymins)
#         zmin = min(zmins)
#         xmax = max(xmaxs)
#         ymax = max(ymaxs)
#         zmax = max(zmaxs)
#     else :
#         xmin = obj.x.min()
#         ymin = obj.y.min()
#         zmin = obj.z.min()
#         xmax = obj.x.max()
#         ymax = obj.y.max()
#         zmax = obj.z.max()

#     # Create the standardized grid
#     x_std = np.linspace(xmin, xmax, 512)
#     y_std = np.linspace(ymin, ymax, 512)
#     z_std = np.linspace(zmin, zmax, 512)

#     # Create a meshgrid for the standardized coordinates
#     x_std_grid, y_std_grid, z_std_grid = np.meshgrid(x_std, y_std, z_std, indexing='ij')
#     img = obj.img
#     # Stack the original coordinates for interpolation
#     points = (obj.x, obj.y, obj.z)

#     # Perform linear interpolation
#     img_std = interpn(points, img, (x_std_grid, y_std_grid, z_std_grid), method='linear', bounds_error=False, fill_value=0)
#     IMG = Image3D.empty(x_std,y_std,z_std)
#     IMG.img = img_std
#     if len(args)==0:
#         return IMG
    
#     out = [IMG]
#     for i,o in enumerate(args) :
#         img = o.img
#         points = (o.x, o.y, o.z)
#         img_std = interpn(points, img, (x_std_grid, y_std_grid, z_std_grid), method='linear', bounds_error=False, fill_value=0)
#         out.append(Image3D.empty(x_std,y_std,z_std))
#         out[i+1].img = img_std
#     return out