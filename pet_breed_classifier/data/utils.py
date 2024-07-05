from torchvision import datasets
from torchvision.transforms import v2

class PerBreedDataModule:
    def __init__(self, dir, batch_size):
        self.dir = dir
        self.batch_size = batch_size
        
    def prepare_data(self):
        datasets.OxfordIIITPet(
            root = "data",
            split = "trainval",
            download = True,
            transform = None
        )
        
    def setup(self):
        train_ds = datasets.Ox