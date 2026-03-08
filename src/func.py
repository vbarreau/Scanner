import numpy as np
import os
import re
import cv2


def compress(img_ini, seuil, fac=0.5):
    img3D = (len(img_ini.shape)==3)
    if img3D:
        img = np.asarray(img_ini)
    else :
        img_out = img_ini.copy()
        img_flat = img_ini[:,3]
        img = img_flat
    mask = img < seuil
    out = np.empty_like(img)
    out[mask] = img[mask]
    x = img[~mask]
    denom = (x - seuil) / (img.max() - seuil + 1e-8)
    out[~mask] = seuil + (x - seuil) * fac / (1 + fac * denom)
    if img3D :
        return out
    else:
        img_out[:,3] = out
        return img_out  

def contrast(img:np.ndarray,val):
    a = (6e-3) * val +1
    if img.ndim == 3:
        M = img.max()
        out = 4*(1-a)/M**2 * img*img*img + 6*(a-1)/M * img*img + (3-2*a)*img
        out[img<0]=0
        return out 
    else:
        M= img[:,3].max()
        out = img.copy()
        out[:,3] = 4*(1-a)/M**2 * img[:,3]*img[:,3]*img[:,3] + 6*(a-1)/M * img[:,3]*img[:,3] + (3-2*a)*img[:,3]
        return out 

def generate_video(image_folder,outpath='scan_anim.mp4'):
    images = [img for img in os.listdir(image_folder) if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    def extract_number(name):
        m = re.search(r'(\d+)', name)
        return int(m.group(1)) if m else float('inf')
    images.sort(key=extract_number)

    # Set frame from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'DIVX'), 5, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

    # ...existing code...

def apply_gradient(img: np.ndarray, axis: int = 0, 
                   start: float = 0.0, end: float = 1.0,
                   inplace: bool = False):
    """
    Apply a 1D gradient along `axis` to a 3D array.

    Parameters
    - img: input 3D array (numpy-compatible)
    - axis: axis along which the gradient is applied (0, 1 or 2)
    - start, end: gradient range (weights will run from start to end)
    - inplace: if True modifies provided numpy array (if writable) and returns it,
               otherwise returns a new array.

    Returns
    - array with gradient applied (same shape as img)
    """
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError("apply_gradient expects a 3D array")

    n = arr.shape[axis]

    # build 1D weight vector
    w = np.linspace(start, end, n, dtype=float)

    # reshape weights to broadcast along img
    shape = [1] * arr.ndim
    shape[axis] = n
    w = w.reshape(shape)
    if inplace:
        # try to modify input array if possible
        arr *= w
        if isinstance(img, np.ndarray):
            # ensure original array updated
            img[:] = arr
        return img
    else:
        return arr * w
      
      
def clip(img, min_val=0.0, max_val=255.0, delete=True):
    """Clip values in img to the range [min_val, max_val].
    If delete=True, values outside the range are set to 0; otherwise they are set to min_val or max_val.
    """
    return np.clip(img, min_val, max_val)

if __name__=='__main__':
    exit()

