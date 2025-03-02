import torch
import hydra
import os.path as osp
import lightning.pytorch as pl
import torch.nn.functional as F

from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dataset import ObjectDataset, GraspDataset
from model.dexGraspEvaluator import DexGraspDetector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@hydra.main(version_base="v1.2", config_path='conf', config_name='best')
def test(cfg: DictConfig) -> None:
    dataset = GraspDataset(is_test=True, object_dataset=ObjectDataset())
    test_loader = DataLoader(dataset,batch_size=32)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
    )
    checkpoint = osp.join("checkpoints", 'best.ckpt')

    model = DexGraspDetector.load_from_checkpoint(checkpoint)
    trainer.test(model, dataloaders=test_loader)

if __name__ == '__main__':
    test()
