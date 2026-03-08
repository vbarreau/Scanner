import numpy as np
from unittest.mock import patch, MagicMock
import os
import sys
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import tranche  # noqa: E402

def make_mock_ds(orientation=[1,0,0, 0,1,0]):
    ds = MagicMock()
    ds.ImageOrientationPatient = orientation
    return ds

def test_normal_axis_z():
    with patch('pydicom.dcmread', return_value=make_mock_ds([1,0,0, 0,1,0])):
        t = tranche.Tranche('fake/path')
        assert t.normal_axis() == 2
