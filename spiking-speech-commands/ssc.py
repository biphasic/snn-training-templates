import pytorch_lightning as pl
from tonic import datasets, transforms, DiskCachedDataset
from torch.utils.data import DataLoader
import tonic
from tonic import transforms


class SSC(pl.LightningDataModule):
    def __init__(
            self, 
            batch_size, 
            encoding_dim=100, 
            dt=4000, 
            num_workers=8,
            data_dir='./data'
        ):
        super().__init__()
        self.save_hyperparameters()


        self.transform = transforms.Compose(
            [
                transforms.Downsample(spatial_factor=encoding_dim / 700),
                transforms.CropTime(max=1e6),
                transforms.ToFrame(
                    sensor_size=(encoding_dim, 1, 1), time_window=dt, include_incomplete=True
                ),
            ]
        )

    def prepare_data(self):
        datasets.SSC(self.hparams.data_dir, split='train')
        datasets.SSC(self.hparams.data_dir, split='valid')
        datasets.SSC(self.hparams.data_dir, split='test')
  
    def setup(self, stage=None):
        cache_path = lambda dir: f"cache/ssc/{dir}/{self.hparams.encoding_dim}/{self.hparams.dt}"
        self.train_data = DiskCachedDataset(
            dataset=datasets.SSC(self.hparams.data_dir, split='train', transform=self.transform),
            cache_path=cache_path("train"),
        )
        self.valid_data = DiskCachedDataset(
            dataset=datasets.SSC(self.hparams.data_dir, split='valid', transform=self.transform),
            cache_path=cache_path("valid"),
        )
        self.test_data = DiskCachedDataset(
            dataset=datasets.SSC(self.hparams.data_dir, split='test', transform=self.transform),
            cache_path=cache_path("test"),
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, 
                          collate_fn=tonic.collation.PadTensors(batch_first=True), shuffle=True, drop_last=True)
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, 
                          collate_fn=tonic.collation.PadTensors(batch_first=True))

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size, 
                          collate_fn=tonic.collation.PadTensors(batch_first=True))
