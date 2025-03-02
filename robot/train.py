import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.dataset import ObjectDataset, GraspDataset
from model.dexGraspEvaluator import DexGraspDetector


@hydra.main(version_base="v1.2", config_path='conf', config_name='default')
def train(cfg: DictConfig) -> None:
    train_dataset = GraspDataset(object_dataset=ObjectDataset())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = DexGraspDetector(**cfg.model)
    logger = pl.loggers.TensorBoardLogger("tb_logs", **cfg.logger)

    # 定义训练器
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=[0],                 # 选择 1 个 GPU (或用列表比如 devices=[0,1] 多卡训练)
        max_epochs=cfg.trainer.max_epochs
    )

    # 训练模型
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == '__main__':
    train()
