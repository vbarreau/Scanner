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

