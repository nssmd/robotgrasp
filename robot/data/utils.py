
import torch
import xml.dom.minidom
import numpy as np
np.float = float
import urdfpy
import open3d as o3d
import os.path as osp
import torch.nn.functional as F
from typing import Union, List, Dict, Tuple


upper_limit = np.asarray([0.6109, 0.4363, 1.57, 1.57, 1.57, 0.4363, 1.5707, 1.5707, 1.5707, 0.4363, 1.5707, 1.5707, 1.5707, 0.6981, 0.4363, 1.5707, 1.5707, 1.5707, 1.047, 1.309, 0.2618, 0.5237, 1.5707])
lower_limit = np.asarray([-0.7854, -0.4363, 0, 0, 0, -0.4363, 0, 0, 0, -0.4363, 0, 0, 0, 0, -0.4363, 0, 0, 0, -1.047, 0, -0.2618, -0.5237, 0])
upper_limit = np.round(upper_limit * 180 / np.pi)
lower_limit = np.round(lower_limit * 180 / np.pi)
joints_middle = (upper_limit + lower_limit) / 2
joints_range = (upper_limit - lower_limit) / 2

class GripperModel:
    """
    This class is for generating a batch of point cloud of the hand surfaces from a batch of grasp poses.
    After initialization, you can use compute_pcd() to generate surface point cloud corresponding to your input grasp pose.
    """
    def __init__(self, gripper_name : str = 'shadow', use_complete_points: bool = False):
        self.sample_point_number = 500
        self.name = gripper_name

        robot = urdfpy.URDF.load(osp.join('asset', 'gripper', f'{gripper_name}.urdf'))
        base_link = robot.links[1]
        links, joints, is_distal = self.select_link_and_joint(robot, base_link)
        link_complete_points, link_complete_normals = self.get_link_pcd_from_mesh(links)
        link_partial_points, link_partial_normals = self.load_points_and_normal_from_xml(osp.join('asset', 'gripper', f'{gripper_name}.xml'), links)
        # for key, value in link_complete_points.items():
        #     print(f"键1: {key}, 值的形状: {value.shape}")
        # for key, value in link_partial_points.items():
        #     print(f"键2: {key}, 值的形状: {value.shape}")
        self.links = links
        self.robot = robot
        self.joints = joints
        self.base_link = base_link
        self.is_distal = is_distal
        self.link_points = link_partial_points
        self.link_normals = link_partial_normals
        if use_complete_points:
            self.link_points = link_complete_points
            self.link_normals = link_complete_normals

    def load_points_and_normal_from_xml(self, xml_file: str, links: List[urdfpy.Link]) -> Tuple[Dict, Dict]:
        name2link = dict()
        points = dict()
        normals = dict()
        DOMTree = xml.dom.minidom.parse(xml_file)
        collection = DOMTree.documentElement
        links_data = collection.getElementsByTagName("PointCloudLinkData")

        for link in links:
            name2link[link.name] = link
            points[link] = np.asarray([])
            normals[link] = np.asarray([])
            collisions = link.collisions
            if len(collisions) == 0:
                continue
        for link_data in links_data:
            name = link_data.getElementsByTagName('linkName')[0].childNodes[0].data
            link = name2link[name]
            point_tmp = list()
            normal_tmp = list()
            point_data = link_data.getElementsByTagName('points')[0].getElementsByTagName('Vector3')
            normal_data = link_data.getElementsByTagName('normal')[0].getElementsByTagName('Vector3')
            for point in point_data:
                x = point.getElementsByTagName('x')[0].childNodes[0].data
                y = point.getElementsByTagName('y')[0].childNodes[0].data
                z = point.getElementsByTagName('z')[0].childNodes[0].data
                point_tmp.append([float(z), -float(x), float(y)])
            for normal in normal_data:
                x = normal.getElementsByTagName('x')[0].childNodes[0].data
                y = normal.getElementsByTagName('y')[0].childNodes[0].data
                z = normal.getElementsByTagName('z')[0].childNodes[0].data
                normal_tmp.append([float(z), -float(x), float(y)])
            points[link] = np.asarray(point_tmp)
            normals[link] = np.asarray(normal_tmp)
        return points, normals

    def compute_pcd(self, Tbase: torch.Tensor = torch.eye(4).unsqueeze(0),
                   joint_para: torch.Tensor = torch.zeros((23, )).unsqueeze(0),
                   compute_weight: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None]]:
        robot = self.robot
        links = self.links
        joints = self.joints
        base_link = self.base_link
        link_points = self.link_points
        link_normals = self.link_normals
        is_distal = self.is_distal
        T = dict()
        T[base_link] = Tbase
        device = joint_para.device
        batch_size = Tbase.shape[0]
        # 遍历字典
        # for key, value in self.link_points.items():
        #     print(f"键: {key}, 值的形状: {value.shape}")
        #
        # print(len(links))
        # compute transform matrix with respect to world coordinate
        for link in links[1:]:
            joint = self.joint_parent[link]
            parent = self.link_parent[link]
            cfg = None

            # judge if the joint is mimic
            if joint.mimic:
                mimic_joint = robot._joint_map[joint.mimic.joint]
                if mimic_joint in joints:
                    cfg = joint_para[:, self.joint2num[mimic_joint]]
                    cfg = joint.mimic.multiplier * cfg + joint.mimic.offset
            elif joint.joint_type != 'fixed':
                 cfg = joint_para[:, self.joint2num[joint]]
            if isinstance(cfg, torch.Tensor):
                cfg = cfg.to(device)
            origin = torch.Tensor(joint.origin).type(Tbase.dtype).to(device)
            if cfg is None or joint.joint_type == 'fixed':
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
            elif joint.joint_type in ['revolute', 'continuous']:
                R = self.r_matrix(cfg, torch.Tensor(joint.axis).to(device))
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
                pose = torch.einsum('ijk,ikl->ijl', pose, R)
            elif joint.joint_type == 'prismatic':
                translation = torch.tile(torch.eye(4)[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
                tmp = torch.einsum('ij, i->ij',
                                   torch.tile(torch.Tensor(joint.axis[np.newaxis, :]).to(device), (batch_size, 1)), cfg)
                translation[:, :3, 3] = tmp
                pose = torch.tile(origin.unsqueeze(0), (batch_size, 1, 1))
                pose = torch.einsum('ijk,ikl->ijl', pose, translation)
            else:
                pose = torch.tile(origin[np.newaxis, :, :], (batch_size, 1, 1)).to(device)
            pose = torch.einsum('ijk,ikl->ijl', T[parent], pose)
            T[link] = pose
        fk = T

        # compute the point cloud of each link
        whole_points = torch.asarray([])
        whole_normals = torch.asarray([])
        whole_weights = torch.asarray([])
        total_points = 0
        for i in range(len(links)):
            link = links[i]
            fk_matrix = fk[link]
            if len(link_points[link]) == 0: continue
            rotation_matrix = fk_matrix[:, :3, :3]
            translation_matrix = fk_matrix[:, :3, 3].unsqueeze(1)
            points = torch.einsum('ijk, ilk->ijl',
                                  torch.tile(torch.Tensor(link_points[link][np.newaxis, :, :]).to(Tbase.dtype), (batch_size, 1, 1)).to(
                                      device), rotation_matrix) + translation_matrix
            normals = torch.einsum('ijk, ilk->ijl', torch.tile(torch.Tensor(link_normals[link][np.newaxis, :, :]).to(Tbase.dtype),
                                                               (batch_size, 1, 1)).to(device), rotation_matrix)
            weights = torch.ones((batch_size, points.shape[1])) * (1 if is_distal[i] else 0.01)
            weights = weights.to(device)
            if len(whole_points) == 0:
                whole_points = points
                whole_normals = normals
                whole_weights = weights
            else:
                whole_points = torch.cat((whole_points, points), dim=1)
                whole_normals = torch.cat((whole_normals, normals), dim=1)
                whole_weights = torch.cat((whole_weights, weights), dim=1)

            total_points += points.shape[1]
            # print(f"Link {link.name}: {points.shape[1]} points added. Total so far: {total_points}")

        if compute_weight:
            return whole_points, whole_normals, whole_weights
        else:
            return whole_points, whole_normals, None

    def r_matrix(self, angle: torch.Tensor, direction: torch.Tensor):
        device = angle.device
        sina = torch.sin(angle).to(device)
        cosa = torch.cos(angle).to(device)
        batch_size = cosa.shape[0]
        direction = direction[:3] / torch.linalg.norm(direction[:3])
        # rotation matrix around unit vector
        M = torch.einsum('ijk,i->ijk', torch.tile(torch.eye(4)[np.newaxis, :, :], (batch_size, 1, 1)).to(device), cosa)
        M[:, 3, 3] = torch.ones_like(cosa)
        M += F.pad(torch.einsum('ijk,i->ijk',
                                torch.tile(torch.outer(direction, direction)[np.newaxis, :, :], (batch_size, 1, 1)),
                                torch.ones_like(cosa) - cosa), (0, 1, 0, 1, 0, 0), 'constant', value=0)
        direction = torch.einsum('ij, i->ij', torch.tile(direction[np.newaxis, :], (batch_size, 1)), sina)
        tmp_matrix = torch.zeros((batch_size, 4, 4)).to(device)
        tmp_matrix[:, 0, 1] = -direction[:, 2]
        tmp_matrix[:, 0, 2] = direction[:, 1]
        tmp_matrix[:, 1, 0] = direction[:, 2]
        tmp_matrix[:, 1, 2] = -direction[:, 0]
        tmp_matrix[:, 2, 0] = -direction[:, 1]
        tmp_matrix[:, 2, 1] = direction[:, 0]
        M += tmp_matrix
        return M

    def get_link_pcd_from_mesh(self, links) -> Tuple[Dict[urdfpy.Link, np.ndarray], Dict[urdfpy.Link, np.ndarray]]:
        scale = 1
        if self.name == 'shadow':
            scale = 0.001

        link_points = dict()
        link_normals = dict()

        for link in links:
            (link_points[link], link_normals[link]) = self.link_point_sample(link, self.sample_point_number)
            link_points[link] *= scale

            if link.name == "panda_rightfinger":
                link_points[link][:, 1] = -link_points[link][:, 1]
        # print(len(link_points))
        return link_points, link_normals

    def link_point_sample(self, link: urdfpy.Link,
                          sample_point_number: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        if len(link.visuals) == 0:
            # print(link.name, "No collision mesh here!")
            return np.asarray([]), np.asarray([])
        geometry = link.visuals[0].geometry
        if geometry.meshes is not None:
            mesh = geometry.meshes[0]
            v = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
            f = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
            mesh = o3d.geometry.TriangleMesh(v, f)
            mesh.compute_vertex_normals()
            PC = mesh.sample_points_uniformly(number_of_points=sample_point_number, use_triangle_normal=True)
            points = np.asarray(PC.points)
            normals = np.asarray(PC.normals)
            # print(len(points))
            return points, normals
        return np.asarray([]), np.asarray([])

    def select_link_and_joint(self, robot: urdfpy.URDF, base_link: urdfpy.Link) -> Tuple[List[urdfpy.Link], List[urdfpy.Joint], List[bool]]:
        link_parent = dict()
        joint_parent = dict()
        joint2num = dict()

        links = list()
        joints = list()
        link_not_distal = set()
        link_not_distal.add(base_link)

        # select the links on subtree under the base_link
        for link in robot.links:
            if base_link in robot._paths_to_base[link]:
                links.append(link)

        # get the parent-child relation between links
        for link in links:
            if link == base_link:
                continue
            path = robot._paths_to_base[link]
            parent_link = path[1]
            joint = robot._G.get_edge_data(link, parent_link)['joint']
            link_parent[link] = parent_link
            link_not_distal.add(parent_link)
            joint_parent[link] = joint
            if joint.mimic is None and joint.joint_type != 'fixed':
                joints.append(joint)
        link_parent[base_link] = None

        # use the Topological Sorting to get a queue of links
        links_queue = list()
        # manual stack
        s = list()
        for link in links:
            if link not in links_queue:
                s.append(link)
                while len(s) > 0:
                    top = s[-1]
                    if not link_parent[top] or link_parent[top] in links_queue:
                        s.pop()
                        links_queue.append(top)
                    else:
                        s.append(link_parent[top])
        for i in range(len(joints)):
            joint2num[joints[i]] = i

        # judge if link is distal
        is_distal = []
        for i in range(len(links_queue)):
            if links_queue[i] not in link_not_distal:
                is_distal.append(True)
            else:
                is_distal.append(False)

        self.link_parent = link_parent
        self.joint_parent = joint_parent
        self.joint2num = joint2num
        return links_queue, joints, is_distal


def normalized_joint_state(joint_state: np.ndarray) -> np.ndarray:
    normalized_state = (joint_state - joints_middle) / joints_range
    normalized_state = np.clip(normalized_state, -1.0, 1.0)
    return normalized_state

def recover_joint_state(normalized_state: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(normalized_state, torch.Tensor):
        device = normalized_state.device
        dtype = normalized_state.dtype
        joint_state = normalized_state * torch.from_numpy(joints_range).to(device).to(dtype) + torch.from_numpy(joints_middle).to(device).to(dtype)
    else:
        joint_state = normalized_state * joints_range + joints_middle
    return joint_state


def compute_pose_from_vector(vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # construct the rotation matrix
    pose_a = vector[:, :3]
    pose_a = pose_a / torch.norm(pose_a, dim = 1).unsqueeze(1)
    pose_b = vector[:, 3: 6]
    pose_b = pose_b - torch.sum(torch.mul(pose_a, pose_b), dim = 1).unsqueeze(1) * pose_a
    pose_b = pose_b / torch.norm(pose_b, dim = 1).unsqueeze(1)
    pose_c = torch.cross(pose_a, pose_b, dim = 1)
    R = torch.stack((pose_a, pose_b, pose_c), dim = 2)

    # construct the transformation matrix
    T = torch.eye(4).repeat((vector.shape[0], 1, 1)).to(vector.device)
    pose_t = vector[:, 6: 9]
    T[:, :3, :3] = R
    T[:, :3, 3] = pose_t

    joint_state = recover_joint_state(vector[:, 9:])

    return T, joint_state

def visualize_gripper_and_object(gripper: GripperModel, vector: torch.Tensor, obj_pcds: torch.Tensor) -> None:
    obj_pcds = obj_pcds.detach().cpu().numpy()
    Tbase, joints_state = compute_pose_from_vector(vector)
    joints_state *= np.pi / 180.0
    points, _, _ = gripper.compute_pcd(Tbase, joints_state)
    for i in range(points.shape[0]):
        pcd = points[i]
        obj_pcd = obj_pcds[i]
        visualize_point_cloud(obj_pcd, pcd.detach().cpu().numpy())
        # print(points.shape[0], i)


import open3d as o3d
import numpy as np
from typing import Union


def visualize_point_cloud(cloud: np.ndarray, sample_point: Union[np.ndarray, None] = None) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Vis', width=1920, height=1080)  # 设置高分辨率窗口

    # 设置点的大小和渲染选项
    opt = vis.get_render_option()
    opt.point_size = 3  # 调整点的大小为 3
    opt.background_color = np.ones((3,))  # 背景设置为白色

    # 创建物体点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.paint_uniform_color([1, 0, 0])  # 点云颜色为红色
    vis.add_geometry(pcd)

    # 如果有采样点，将其添加到可视化中
    if sample_point is not None:
        pt = o3d.geometry.PointCloud()
        pt.points = o3d.utility.Vector3dVector(sample_point)
        pt.paint_uniform_color([0, 1, 0])  # 采样点颜色为绿色
        vis.add_geometry(pt)

    vis.run()
    vis.destroy_window()
