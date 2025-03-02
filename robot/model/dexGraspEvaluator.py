import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SimplePointNetPlusPlus, compute_force_closure


class DexGraspDetector(pl.LightningModule):
    def __init__(
            self,
            friction_coeff: float = 0.5,
            num_friction_directions: int = 8,
            point_features_dim: int = 256,
            linear_transform_dim: int = 64,
            combined_features_dim: int = 256,
            attention_dim: int = 128,
            lr: float = 1e-4
    ):
        super(DexGraspDetector, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.friction_coeff = friction_coeff
        self.num_friction_directions = num_friction_directions

        self.point_net = SimplePointNetPlusPlus(feature_dim=point_features_dim)

        self.linear_pose = nn.Sequential(
            nn.Linear(16, linear_transform_dim),
            nn.ReLU(),
            nn.Linear(linear_transform_dim, linear_transform_dim),
            nn.ReLU()
        )

        self.linear_joint = nn.Sequential(
            nn.Linear(23, linear_transform_dim),
            nn.ReLU(),
            nn.Linear(linear_transform_dim, linear_transform_dim),
            nn.ReLU()
        )


        self.attention = nn.Sequential(
            nn.Linear(1, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 3)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(point_features_dim + linear_transform_dim * 2, combined_features_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(combined_features_dim, 1)
        )

        self.test_output = {'loss': [], 'acc': []}

    def forward(
            self,
            pose: torch.Tensor,
            joint: torch.Tensor,
            pcd: torch.Tensor,
            contact_points: torch.Tensor,
            contact_normals: torch.Tensor
    ) -> torch.Tensor:
        B = pose.size(0)

        # 1. 力闭合性计算
        force_closure_scores = self.compute_force_closure(
            contact_points=contact_points,
            contact_normals=contact_normals
        )

        # 2. PointNet++ 处理点云
        point_features = self.point_net(pcd)

        # 3. 矩阵线性变换
        pose_flat = pose.view(B, -1)  # (B, 16)
        pose_features = self.linear_pose(pose_flat)

        joint_features = self.linear_joint(joint)

        # 4. 生成注意力权重
        attention_weights = self.attention(force_closure_scores)
        attention_weights = F.softmax(attention_weights, dim=1)

        # 分割注意力权重
        att_point, att_pose, att_joint = torch.split(attention_weights, 1, dim=1)

        # 5. 应用注意力权重
        point_features = point_features * att_point
        pose_features = pose_features * att_pose
        joint_features = joint_features * att_joint

        # 6. 特征融合
        combined = torch.cat(
            [point_features, pose_features, joint_features],
            dim=1
        )

        # 7. 全连接输出
        pred = self.fc_layers(combined)
        pred = torch.sigmoid(pred).squeeze(1)

        return pred

    def compute_force_closure(
            self,
            contact_points: torch.Tensor,
            contact_normals: torch.Tensor
    ) -> torch.Tensor:
        force_closure_scores = compute_force_closure(
            friction_coeff=self.friction_coeff,
            num_friction_directions=self.num_friction_directions,
            contact_points=contact_points,
            contact_normals=contact_normals
        )
        return force_closure_scores

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:

        pose, joint, pcd, label, contact_points, contact_normals = batch
        label = label.float()

        pred = self.forward(pose, joint, pcd, contact_points, contact_normals)
        label = label.squeeze(1)
        loss = F.binary_cross_entropy(pred, label)

        acc = ((pred > 0.5) == (label > 0.5)).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        pose, joint, pcd, label, contact_points, contact_normals = batch
        pred = self.forward(pose, joint, pcd, contact_points, contact_normals)
        label = label.float()
        label = label.squeeze(1)
        loss = F.binary_cross_entropy(pred, label)
        accuracy = ((pred > 0.5) == (label > 0.5)).float().mean()

        self.test_output['loss'].append(loss.item())
        self.test_output['acc'].append(accuracy.item())

    def on_test_end(self) -> None:
        print('The BCE Loss is: ', np.mean(self.test_output['loss']))
        print('The Accuracy is: ', np.mean(self.test_output['acc']))

    def configure_optimizers(self) -> torch.optim.Optimizer:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
