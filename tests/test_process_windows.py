"""Process selection excludes every soma from the entire physical ROI."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'probe-em' / 'scripts'))
from select_process_windows import process_candidates, soma_regions


def test_entire_process_box_avoids_both_somata_with_nonzero_offset():
    x,y,z = np.indices((128,64,64))
    labels = np.zeros(x.shape,np.uint32)
    labels[((x-24)**2+(y-20)**2+(z-32)**2 <= 12**2)
           | ((x >= 24)&(x <= 114)&((y-20)**2+(z-32)**2 <= 3**2))] = 1
    labels[((x-90)**2+(y-44)**2+(z-32)**2 <= 12**2)
           | ((x >= 12)&(x <= 90)&((y-44)**2+(z-32)**2 <= 3**2))] = 2
    resolution, offset = np.array([200,200,200]),np.array([11,7,3])
    roi_shape, roi_res = np.array([8,8,5]),np.array([200,200,200])
    bounds = [offset*resolution,(offset+labels.shape)*resolution]
    candidates, report = process_candidates(labels,resolution,offset,[1,2],[8,12],roi_shape,roi_res,bounds)
    somata,_,_ = soma_regions(labels,resolution,[1,2])
    assert candidates.keys() == {1,2}
    for label, items in candidates.items():
        for item in items:
            center = np.array(item['center_nm'])
            low = np.floor((center-roi_shape*roi_res/2)/resolution-offset).astype(int)
            high = np.ceil((center+roi_shape*roi_res/2)/resolution-offset).astype(int)
            assert not somata[tuple(slice(a,b) for a,b in zip(low,high))].any()
            assert 350 <= item['locator_radius_nm'] <= 850


def test_no_resolved_soma_is_an_error_instead_of_a_soma_window():
    labels = np.ones((4,4,4),np.uint32)
    with pytest.raises(ValueError,match='No resolved thick soma'):
        soma_regions(labels,[200,200,200],[1])
