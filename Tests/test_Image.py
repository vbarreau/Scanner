import os
import sys
import numpy as np
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import Image  # noqa: E402

    
def test_empty():
    img = Image.Image3D([])
    assert img.shape == (0,0,0)
    
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    z = np.linspace(0, 1, 10)
    img_zero = Image.Image3D.empty(x,y,z)
    assert img_zero.shape == (10,10,10)
    assert (img_zero.img == 0).all()
    assert img_zero.x == x
    assert img_zero.y == y
    assert img_zero.z == z
    
    img_zero.img[5,5,5] = 1
    assert img_zero.fetch(0.5,0.5,0.5) == 1


    