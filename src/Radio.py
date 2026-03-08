import pydicom as pdc
from pydicom.dataset import FileDataset
import matplotlib.pyplot as plt


class Radio(FileDataset):
    def __init__(self, filepath):
        ds = pdc.dcmread(filepath)
        super().__init__(self,ds)
        self.filepath = filepath
    
    def plot(self, ax = None) :
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(pdc.pixel_array(self), cmap='gray')
        ax.set_title(f'{self.SOPInstanceUID}')
        ax.axis('off')


