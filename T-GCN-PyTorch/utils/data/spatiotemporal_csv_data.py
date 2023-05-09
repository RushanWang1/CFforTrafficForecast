import argparse
import numpy as np
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
import utils.data.functions


class SpatioTemporalCSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        feat_path: str,
        adj_path: str,
        numlane_path: str,
        speedlimit_path: str,        
        # weather_path: str,
        # calender_path: str,
        poi_path:str,
        # parking_path:str,
        # gas_path:str,
        # restaurant_path:str,
        temp_path: str,
        precip_path: str,
        wind_path: str,
        humidity_path: str,
        day0_path: str,
        day1_path: str,
        day2_path: str,
        day3_path: str,
        day4_path: str,
        day5_path: str,
        day6_path: str,
        hour_sin_path: str,
        hour_cos_path: str,
        batch_size: int = 24,
        seq_len: int = 12,
        pre_len: int = 12,
        split_ratio: float = 0.8,
        normalize: bool = True,
        **kwargs
    ):
        super(SpatioTemporalCSVDataModule, self).__init__()
        self._feat_path = feat_path
        self._adj_path = adj_path
        # self._weather_path = weather_path
        # self._calender_path = calender_path        
        self._numlane_path = numlane_path
        self._speedlimit_path = speedlimit_path
        self._poi_path = poi_path
        # self._parking_path = parking_path
        # self._gas_path = gas_path
        # self._restaurant_path = restaurant_path
        self._temp_path = temp_path
        self._precip_path = precip_path
        self._wind_path = wind_path
        self._humidity_path = humidity_path
        self._day0_path = day0_path
        self._day1_path = day1_path
        self._day2_path = day2_path
        self._day3_path = day3_path
        self._day4_path = day4_path
        self._day5_path = day5_path
        self._day6_path = day6_path
        self._hour_sin_path = hour_sin_path
        self._hour_cos_path = hour_cos_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_ratio = split_ratio
        self.normalize = normalize
        self._feat = utils.data.functions.load_features(self._feat_path)
        self._feat_max_val = np.max(self._feat)
        self._adj = utils.data.functions.load_adjacency_matrix(self._adj_path)
        # self._weather = utils.data.functions.load_features(self._weather_path)
        # self._calender = utils.data.functions.load_features(self._calender_path)
        self._numlane = utils.data.functions.load_features(self._numlane_path)
        self._speedlimit = utils.data.functions.load_features(self._speedlimit_path)
        self._poi = utils.data.functions.load_features(self._poi_path)
        # self._parking = utils.data.functions.load_features(self._parking_path)
        # self._gas = utils.data.functions.load_features(self._gas_path)
        # self._restaurant = utils.data.functions.load_features(self._restaurant_path)
        self._temp = utils.data.functions.load_features(self._temp_path)
        self._precip = utils.data.functions.load_features(self._precip_path)
        self._wind = utils.data.functions.load_features(self._wind_path)
        self._humidity = utils.data.functions.load_features(self._humidity_path)
        self._day0 = utils.data.functions.load_features(self._day0_path)
        self._day1 = utils.data.functions.load_features(self._day1_path)
        self._day2 = utils.data.functions.load_features(self._day2_path)
        self._day3 = utils.data.functions.load_features(self._day3_path)
        self._day4 = utils.data.functions.load_features(self._day4_path)
        self._day5 = utils.data.functions.load_features(self._day5_path)
        self._day6 = utils.data.functions.load_features(self._day6_path)
        self._hour_sin = utils.data.functions.load_features(self._hour_sin_path)
        self._hour_cos = utils.data.functions.load_features(self._hour_cos_path)

    @staticmethod
    def add_data_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--seq_len", type=int, default=12)
        parser.add_argument("--pre_len", type=int, default=12)
        parser.add_argument("--split_ratio", type=float, default=0.8)
        parser.add_argument("--normalize", type=bool, default=True)
        return parser

    def setup(self, stage: str = None):
        (
            self.train_dataset,
            self.val_dataset,
        ) = utils.data.functions.generate_torch_datasets(
            self._feat,
            self._numlane,
            self._speedlimit,
            self._poi,
            # self._parking,
            # self._gas,
            # self._restaurant,
            self._temp,
            self._precip,
            self._wind,
            self._humidity,
            self._day0,
            self._day1,
            self._day2,
            self._day3,
            self._day4,
            self._day5,
            self._day6,
            self._hour_sin,
            self._hour_cos,
            # self._weather,
            # self._calender,
            # self._POI,
            self.seq_len,
            self.pre_len,
            split_ratio=self.split_ratio,
            normalize=self.normalize,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

    @property
    def feat_max_val(self):
        return self._feat_max_val

    @property
    def adj(self):
        return self._adj
