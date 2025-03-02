import os
import os.path as osp
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import GripperModel
from .utils import recover_joint_state
from .utils import visualize_point_cloud


class ObjectDataset(Dataset):
    def __init__(self,
                 path: str = osp.join('asset', 'data', 'object'),
                 point_num: int = 2048) -> None:
        dirs = os.listdir(path)
        subdirs = [d for d in dirs if osp.isdir(osp.join(path, d))]

        self.point_num = point_num
        self.data = subdirs
        self.rootpath = path

    def __getitem__(self, idx) -> np.ndarray:
        return self.fetch_pointcloud(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)

    def fetch_pointcloud(self, objpath: str) -> np.ndarray:
        objfile = osp.join(self.rootpath, objpath, 'part_meshes', 'complete.ply')
        pcd = o3d.io.read_point_cloud(objfile)
        pcd = np.asarray(pcd.points)

        rand_idx = np.random.permutation(pcd.shape[0])
        pcd = pcd[rand_idx[: self.point_num]]

        return pcd


class GraspDataset(Dataset):
    def __init__(
            self,
            path: str = osp.join('asset', 'data', 'grasp'),
            is_test: bool = False,
            object_dataset: 'ObjectDataset' = None,
            contact_npz_path: str = 'contact_points_train.npz',
            max_contact_points: int = 1000
    ) -> None:

        assert object_dataset is not None, "object_dataset cannot be None."

        if is_test:
            path = osp.join(path, 'test')
            contact_npz_path = 'contact_points_test.npz'
        else:
            path = osp.join(path, 'train')
            contact_npz_path = 'contact_points_train.npz'

        self.object_dataset = object_dataset
        self.samples = []  # 列表，用于存储所有抓取样本
        self.object_indices = []  # 列表，用于存储对应的物体索引
        self.max_contact_points = max_contact_points  # 最大接触点数量

        for grasp_dir in os.listdir(path):
            grasp_dir_path = osp.join(path, grasp_dir)
            if not osp.isdir(grasp_dir_path):
                continue

            obj_idx = self.object_dataset.data.index(grasp_dir)

            # 加载 pose.npy, joint_state.npy, label.npy
            pose_file = osp.join(grasp_dir_path, 'pose.npy')
            joint_file = osp.join(grasp_dir_path, 'joint_state.npy')
            label_file = osp.join(grasp_dir_path, 'label.npy')

            if not (osp.exists(pose_file) and osp.exists(joint_file) and osp.exists(label_file)):
                print(f"Missing files in {grasp_dir_path}, skipping.")
                continue

            poses = np.load(pose_file)
            joints = np.load(joint_file)
            labels = np.load(label_file)

            num_samples = poses.shape[0]
            if labels.ndim == 1:
                labels = labels[:, np.newaxis]
            # 批量添加样本
            self.samples.append({
                'pose': poses,                # (2520, 4, 4)
                'joint': joints,              # (2520, 23)
                'label': labels.astype(float),# (2520, 1)
                'object_idx': obj_idx         # 单个 int
            })

            self.object_indices.extend([obj_idx] * num_samples)

        if self.samples:
            all_poses = np.concatenate([s['pose'] for s in self.samples], axis=0)      # (Total, 4, 4)
            all_joints = np.concatenate([s['joint'] for s in self.samples], axis=0)    # (Total, 23)
            all_labels = np.concatenate([s['label'] for s in self.samples], axis=0)    # (Total, 1)

            self.poses = all_poses
            self.joints = all_joints
            self.labels = all_labels
        else:
            self.poses = np.empty((0, 4, 4))
            self.joints = np.empty((0, 23))
            self.labels = np.empty((0, 1))

        # 加载预计算的接触点和法向量
        if osp.exists(contact_npz_path):
            contact_data = np.load(contact_npz_path)
            self.contact_points = contact_data['contact_points']
            self.contact_normals = contact_data['contact_normals']
        else:
            print(f"Contact points file {contact_npz_path} not found. Precomputing...")
            precompute_contact_points_and_save(
                grasp_dataset=self,
                save_path=contact_npz_path,
                num_contact_points=self.max_contact_points
            )
            contact_data = np.load(contact_npz_path)
            self.contact_points = contact_data['contact_points']
            self.contact_normals = contact_data['contact_normals']

        print(f"Total grasp samples collected: {len(self.object_indices)}")
        if len(self.object_indices) > 0:
            print(f"First sample pose shape: {self.poses[0].shape}, joint shape: {self.joints[0].shape}, label: {self.labels[0]}")
            print(f"Contact points shape: {self.contact_points.shape}, Contact normals shape: {self.contact_normals.shape}")

    def __len__(self) -> int:
        return len(self.object_indices)

    def __getitem__(self, idx) -> tuple:

        pose = self.poses[idx]
        joint = self.joints[idx]
        label = self.labels[idx]
        obj_idx = self.object_indices[idx]
        pcd = self.object_dataset[obj_idx]
        pcd = pcd.astype(np.float32)

        if not hasattr(self, 'contact_points'):
            return pose, joint, pcd, label,None,None

        contact_points = self.contact_points[idx][:self.max_contact_points]
        contact_normals = self.contact_normals[idx][:self.max_contact_points]

        return pose, joint, pcd, label, contact_points, contact_normals

    def compute_contact_points(self, idx: int, num_contact_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        pose, joint, obj_pcd, label ,_,_= self[idx]
        gripper = GripperModel(gripper_name='shadow', use_complete_points=True)

        joint = recover_joint_state(joint) * (np.pi / 180.0)
        Tbase = torch.from_numpy(pose).float().unsqueeze(0)
        joint_para = torch.from_numpy(joint).float().unsqueeze(0)

        gripper_points, gripper_normals, _ = gripper.compute_pcd(Tbase, joint_para,
                                                                 compute_weight=True)

        gripper_points_np = gripper_points[0].detach().cpu().numpy()
        gripper_normals_np = gripper_normals[0].detach().cpu().numpy()

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(obj_pcd)
        distances, indices = nbrs.kneighbors(gripper_points_np)

        distances = distances.flatten()

        sorted_indices = np.argsort(distances)

        # 选择前 num_contact_points 个接触点
        top_indices = sorted_indices[:num_contact_points]

        contact_points = gripper_points_np[top_indices]
        contact_normals = gripper_normals_np[top_indices]

        return contact_points, contact_normals


def precompute_contact_points_and_save(
        grasp_dataset: GraspDataset,
        save_path: str = 'contact_points_test.npz',
        num_contact_points: int = 1000
) -> None:

    contact_points_list = []
    contact_normals_list = []


    for i in tqdm(range(len(grasp_dataset))):
        contact_points, contact_normals = grasp_dataset.compute_contact_points(
            idx=i,
            num_contact_points=num_contact_points
        )

        contact_points_list.append(contact_points)
        contact_normals_list.append(contact_normals)

    contact_points_all = np.stack(contact_points_list, axis=0)
    contact_normals_all = np.stack(contact_normals_list, axis=0)

    np.savez_compressed(
        save_path,
        contact_points=contact_points_all,
        contact_normals=contact_normals_all,
    )
    print(f"All contact points have been saved to {save_path}.")
