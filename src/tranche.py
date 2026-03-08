import numpy as np
import pydicom as pdc
from pydicom.dataset import FileDataset


class Tranche(FileDataset):
    def __init__(self, filepath):
        ds = pdc.dcmread(filepath)
        super().__init__(self, ds,
                         preamble=getattr(ds, 'preamble', None),
                         file_meta=getattr(ds, 'file_meta', None))
        self.filepath = filepath
        self.normale = self.normal_axis()
        

    def vec_plan(self):
        orientation = self.ImageOrientationPatient
        v1 = np.array([orientation[0], orientation[1], orientation[2]])
        v2 = np.array([orientation[3], orientation[4], orientation[5]])
        return v1, v2


    def normal_axis(self):
        v1,v2 = self.vec_plan()
        vn = np.cross(v1, v2)
        x,y,z = vn
        if x !=0 :
            return 0
        elif y != 0:
            return 1
        elif z != 0:
            return 2
        
    @classmethod
    def field_values(cls, filepaths, keyword):
        """Print all distinct values for a given tag keyword across files."""
        import pydicom as pdc
        seen = {}
        for p in filepaths:
            ds = pdc.dcmread(p)
            tag = ds.data_element(keyword)
            val = str(tag.value) if tag is not None else None
            if val not in seen:
                seen[val] = p
        for val, path in seen.items():
            print(f'{path}: {val}')

    @classmethod
    def diff_fields(cls, filepaths):
        """Return the set of tag keyword names whose values differ across files."""
        import pydicom as pdc
        datasets = [pdc.dcmread(p) for p in filepaths]
        # Collect all tags present in at least one file (excluding pixel data)
        all_tags = set()
        for ds in datasets:
            for elem in ds:
                if elem.tag != (0x7FE0, 0x0010):  # skip Pixel Data
                    all_tags.add(elem.tag)
        differing = set()
        for tag in all_tags:
            values = []
            for ds in datasets:
                if tag in ds:
                    try:
                        values.append(str(ds[tag].value))
                    except Exception:
                        values.append(None)
                else:
                    values.append(None)
            if len(set(values)) > 1:
                try:
                    keyword = datasets[0][tag].keyword if tag in datasets[0] else str(tag)
                except Exception:
                    keyword = str(tag)
                differing.add(keyword or str(tag))
        return differing

    def compare(self,other):
        gen_self = self.iterall()
        gen_other = other.iterall()
        n = 0
        str_output = ''
        try :
            while True :
                att_self = next(gen_self)
                att_other = next(gen_other)
                if att_self != att_other:
                    str_output += f'\n{att_self}\n{att_other}\n'
                    n+=1
        except(StopIteration):
            # print(f'{n} différences')
            return(str_output,n)

