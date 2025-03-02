
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.dataset import GraspDataset, ObjectDataset

# -----------------------------
# 1. 接触点与力闭合相关函数
# -----------------------------
def compute_tangents(contact_normals: torch.Tensor) -> torch.Tensor:

    B, M, _ = contact_normals.shape
    device = contact_normals.device
    tangents = torch.zeros((B, M, 2, 3), device=device)


    arbitrary = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 1, 3)
    cross = torch.cross(contact_normals, arbitrary, dim=-1)
    cross_norm = torch.norm(cross, dim=-1, keepdim=True)
    cross = cross / (cross_norm + 1e-6)

    cross2 = torch.cross(contact_normals, cross, dim=-1)
    cross2 = F.normalize(cross2, dim=-1)

    tangents[:, :, 0, :] = cross
    tangents[:, :, 1, :] = cross2

    return tangents


def compute_contact_points_model(
        pose: torch.Tensor,
        num_contact_points: int = 100
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    B = pose.size(0)
    contact_points_list = []
    contact_normals_list = []
    obj_dataset = ObjectDataset()
    dataset = GraspDataset(object_dataset=obj_dataset, is_test=False)
    for i in range(B):
        cp, cn = dataset.compute_contact_points(
            i,
            num_contact_points=num_contact_points
        )
        contact_points_list.append(cp)
        contact_normals_list.append(cn)

    return contact_points_list, contact_normals_list


def compute_force_closure(
    friction_coeff: float,
    num_friction_directions: int,
    contact_points: torch.Tensor,
    contact_normals: torch.Tensor
) -> torch.Tensor:
    B, M, _ = contact_points.shape
    device = contact_points.device

    if M == 0:
        return torch.zeros((B, 1), device=device)

    # 1. 计算切向向量
    tangents = compute_tangents(contact_normals)

    # 2. 生成 D 个摩擦方向的角度
    D = num_friction_directions
    angles = torch.linspace(0, 2 * np.pi, D, device=device)
    angles = angles.view(1, 1, D, 1)

    # 3. 计算摩擦方向
    theta = torch.atan(torch.tensor(friction_coeff, device=device))
    n = contact_normals.unsqueeze(2)
    t1 = tangents[:, :, 0, :].unsqueeze(2)
    t2 = tangents[:, :, 1, :].unsqueeze(2)

    # 在圆锥面上均匀取 D 条切向
    f_d = t1 * torch.cos(angles) + t2 * torch.sin(angles) + n * torch.cos(theta)
    f_d = F.normalize(f_d, dim=-1)

    # 4. 算协方差矩阵
    forces = f_d.view(B, -1, 3)
    cov = torch.bmm(forces.transpose(1, 2), forces) / (M * D)
    det = torch.det(cov).unsqueeze(1)
    # 5. 取 ReLU
    fc = F.relu(det)

    return fc

# -----------------------------
# 2. PointNet++ 相关
# -----------------------------
def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:

    B, N, C = src.shape
    _, M, _ = dst.shape
    dist = -2.0 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:

    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def query_ball_point(
    radius: float,
    nsample: int,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor
) -> torch.Tensor:

    device = xyz.device
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]

    sqrdists = square_distance(new_xyz, xyz)

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    group_idx[sqrdists > radius**2] = N

    group_idx = group_idx.sort(dim=-1)[0][..., :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:

    device = points.device
    B = points.shape[0]

    if idx.ndim == 2:
        S = idx.shape[1]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).repeat(1, S)
        new_points = points[batch_indices, idx, :]
    elif idx.ndim == 3:
        S, K = idx.shape[1], idx.shape[2]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1).repeat(1, S, K)
        new_points = points[batch_indices, idx, :]
    else:
        raise ValueError(f"idx shape {idx.shape} is wrong")

    return new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self,
        npoint: int,
        radius: float,
        nsample: int,
        in_channel: int,
        mlp: list,
        group_all: bool
    ):

        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        last_channel = in_channel
        self.conv_blocks = nn.ModuleList()
        for out_channel in mlp:
            block = nn.Sequential(
                nn.Conv2d(last_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )
            self.conv_blocks.append(block)
            last_channel = out_channel

    def forward(self, xyz: torch.Tensor, points: torch.Tensor):

        device = xyz.device
        B, N, _ = xyz.shape

        if self.group_all or self.npoint is None:

            new_xyz = torch.zeros(B, 1, 3, device=device)
            grouped_xyz = xyz.view(B, 1, N, 3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:

            idx_fps = farthest_point_sampling(xyz, self.npoint)
            new_xyz = index_points(xyz, idx_fps)


            idx_ball = query_ball_point(self.radius, self.nsample, xyz, new_xyz)


            grouped_xyz = index_points(xyz, idx_ball)

            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx_ball)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz

        new_points = new_points.permute(0, 3, 2, 1)

        for conv in self.conv_blocks:
            new_points = conv(new_points)

        new_points = F.max_pool2d(new_points, kernel_size=[new_points.shape[2], 1])
        new_points = new_points.squeeze(2)

        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class SimplePointNetPlusPlus(nn.Module):

    def __init__(self, feature_dim=256):
        super(SimplePointNetPlusPlus, self).__init__()


        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3,
            mlp=[32, 32, 64],
            group_all=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=64 + 3,
            mlp=[64, 64, 128],
            group_all=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=True
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor=None) -> torch.Tensor:

        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # l3_points => (B,1,256) => (B,256)
        x = l3_points.view(-1, 256)

        x = self.fc_layers(x)
        return x