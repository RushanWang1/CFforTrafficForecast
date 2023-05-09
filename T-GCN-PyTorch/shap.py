import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
# import utils.email
import utils.logging
import torch
import shap

DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
    "ventura": {"feat": "TGCN/T-GCN-PyTorch/data/speed_few_T.csv", "adj": "TGCN/T-GCN-PyTorch/data/adjacent_matrix_G_dual_few.csv"},
}

dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS["ventura"]["feat"], adj_path=DATA_PATHS["ventura"]["adj"], 
        numlane_path = "TGCN/T-GCN-PyTorch/data/lane_few_feature.csv",
        speedlimit_path = "TGCN/T-GCN-PyTorch/data/speedlimit_few_feature.csv",
        POI_path="TGCN/T-GCN-PyTorch/data/osmPOI_feature_few.csv",
        temp_path = "TGCN/T-GCN-PyTorch/data/temp_feature.csv",
        precip_path = "TGCN/T-GCN-PyTorch/data/precip_feature.csv",
        wind_path = "TGCN/T-GCN-PyTorch/data/wind_feature.csv",
        humidity_path = "TGCN/T-GCN-PyTorch/data/humidity_feature.csv",        
        day_path="TGCN/T-GCN-PyTorch/data/dayofweek_feature.csv",
        hour_path="TGCN/T-GCN-PyTorch/data/hourofday_feature.csv",        
        #  **vars(args)
    )
# model = get_model(args, dm)
# task = get_task(args, model, dm)
# callbacks = get_callbacks(args)
# trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
# trainer.fit(task, dm)
# results = trainer.validate(datamodule=dm)
mymodel = models.TGCN(adj=dm.adj, hidden_dim=64)
checkpoint = torch.load("model/lightning_logs/version_15306969\checkpoints\epoch=7-step=432.ckpt", map_location=torch.device('cpu'))
sd = mymodel.state_dict()
model = mymodel.load_state_dict(sd)
# model = mymodel.load_from_checkpoint("model/lightning_logs/version_15306969\checkpoints\epoch=7-step=432.ckpt")

batch = next(iter(dm.val_dataloader))
x, _, _ = batch
background = x[:100].to(model.device)
test_points = x[100:180].to(model.device)
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(test_points)
