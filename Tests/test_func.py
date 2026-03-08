import os
import sys
import numpy as np
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import func  # noqa: E402

def test_compress():
    # compress supports 3D arrays or 2D arrays of shape (N, 4)

    # --- 3D case ---
    img3d = np.array([[[10.0, 200.0]]])  # shape (1,1,2)
    seuil = 50.0
    out3d = func.compress(img3d, seuil, fac=0.5)
    assert out3d.shape == img3d.shape
    assert out3d[0, 0, 0] == 10.0    # below threshold: unchanged
    assert out3d[0, 0, 1] < 200.0   # above threshold: compressed

    # Larger 3D array
    img = np.array([[[10.0, 50.0, 100.0, 200.0]]])
    out = func.compress(img, seuil, fac=0.5)
    assert out[0, 0, 0] == 10.0     # below: unchanged
    assert out[0, 0, 1] == 50.0     # at threshold: unchanged
    assert out[0, 0, 2] < 100.0    # above: compressed
    assert out[0, 0, 3] < 200.0    # above: compressed

    # --- 2D (N, 4) case: only column 3 is compressed ---
    img2d = np.array([[0.0, 0.0, 0.0, 200.0],
                      [0.0, 0.0, 0.0,  10.0]])
    out2d = func.compress(img2d, seuil, fac=0.5)
    assert out2d.shape == img2d.shape
    assert out2d[0, 3] < 200.0     # above threshold: compressed
    assert out2d[1, 3] == 10.0     # below threshold: unchanged


def test_contrast():
    img = np.ones((4, 4, 4)) * 100.0

    # val=0 → a=1 → cubic reduces to identity: 4(1-1)/M²·x³ + 6(1-1)/M·x² + (3-2)·x = x
    out_zero = func.contrast(img, 0)
    np.testing.assert_allclose(out_zero, img, rtol=1e-6)

    # Output shape is preserved
    img_ramp = np.linspace(0, 1000, 8).reshape(2, 2, 2)
    out_high = func.contrast(img_ramp, 50)
    assert out_high.shape == img_ramp.shape

    # The cubic preserves 0, M/2, and M as fixed points, but reshapes the curve.
    # Test with 750 (above midpoint of [0, 1000]):
    # val>0 (a>1): values above midpoint get boosted further toward M
    # val<0 (a<1): values above midpoint get pulled back toward midpoint
    img_test = np.array([[[0.0, 750.0, 1000.0]]])  # max=1000, midpoint=500
    out_pos = func.contrast(img_test, 50)
    out_neg = func.contrast(img_test, -50)
    assert out_pos[0, 0, 1] > 750.0   # val>0: value above midpoint boosted
    assert out_neg[0, 0, 1] < 750.0   # val<0: value above midpoint reduced


def test_apply_gradient():
    img = np.ones((4, 6, 8))

    # Non-3D input raises ValueError
    try:
        func.apply_gradient(np.ones((4, 4)), axis=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    # Along axis 0: first slice weight=start, last slice weight=end
    out = func.apply_gradient(img, axis=0, start=0.0, end=1.0)
    assert out.shape == img.shape
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[-1], 1.0)

    # Along axis 1
    out1 = func.apply_gradient(img, axis=1, start=0.2, end=0.8)
    np.testing.assert_allclose(out1[:, 0, :], 0.2)
    np.testing.assert_allclose(out1[:, -1, :], 0.8)

    # Along axis 2
    out2 = func.apply_gradient(img, axis=2, start=1.0, end=2.0)
    np.testing.assert_allclose(out2[:, :, 0], 1.0)
    np.testing.assert_allclose(out2[:, :, -1], 2.0)

    # inplace=True modifies the original array
    arr = np.ones((3, 3, 3))
    func.apply_gradient(arr, axis=0, start=0.0, end=1.0, inplace=True)
    np.testing.assert_allclose(arr[0], 0.0)
    np.testing.assert_allclose(arr[-1], 1.0)

