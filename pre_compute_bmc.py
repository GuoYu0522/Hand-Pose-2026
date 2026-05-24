import os

import numpy as np
from tqdm import tqdm

import BioMC.config_bmc_loss as cfg_bmc
import BioMC.cal_joint_angles as cal_joint_angles


def load_joints_(set, mode):
    path = os.path.join("BMC", "joints", "{}_{}.npy".format(set, mode))
    joints = np.load(path)
    return joints


def normalize(vec):
    len = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / len


def load_joints(dict):
    joints_all = []
    for set, mode in dict.items():
        for m in mode:
            path = os.path.join("BMC", "joints", "{}_{}.npy".format(set, m))
            if not os.path.exists(path):
                raise FileNotFoundError(f"File NOT Exist: {path}")
            joints = np.load(path)
            joints_all.append(joints)
    joints_all = np.concatenate(joints_all, axis=0).astype(np.float32)  # (N, 21,3)
    return joints_all


if __name__ == '__main__':

    path = "BMC"
    if not os.path.isdir(path):
        os.makedirs(path)

    # original 3D joints in minimal-hand's setting
    th_dict = {"rhd": ["train", "test"], "gan": ["train"]}
    th_joints = load_joints(th_dict)
    joints = th_joints.copy()  # (N, 21, 3)

    # all joints available setting
    # all_dict = {"rhd": ["train", "test"], "gan": ["train"], "stb": ["train", "test"], "fh": ["train"]}
    # all_joints = load_joints(all_dict)
    # joints = all_joints.copy()  # (N, 21, 3)

    # root-relative
    joints_root = np.expand_dims(joints[:, cfg_bmc.JOINT_ROOT_IDX, :], 1)
    joints = joints - joints_root

    # scale-invariant
    ref_bones = np.linalg.norm(joints[:, cfg_bmc.REF_BONE_LINK[0]] - joints[:, cfg_bmc.REF_BONE_LINK[1]], axis=1)
    ref_bones = np.expand_dims(ref_bones, [1, 2])
    joints = joints / ref_bones

    # loop
    # for i in tqdm(range(0, joints.shape[0], 5000)):
    #     joints_i = joints[i]
    #     # visualization
    #     vis.plot3d(joints_i)
    #     bones = bone.caculate_length(joints_i, label="all")
    # print(bones[9])

    # calculate bone length limits
    kin_chain = [
        joints[:, i] - joints[:, cfg_bmc.SNAP_PARENT[i]]
        for i in range(1, 21)
    ]
    kin_chain = np.array(kin_chain)
    kin_chain = kin_chain.swapaxes(1, 0)  # (N*20*3)

    bone_lens = np.linalg.norm(
        kin_chain, ord=2, axis=-1, keepdims=True
    )  # (N*20)
    bone_lens = np.squeeze(bone_lens)

    max_bone_len = np.max(bone_lens, axis=0)  # (20,)
    min_bone_len = np.min(bone_lens, axis=0)  # (20,)
    np.save("BMC/bone_len_min.npy", min_bone_len)
    np.save("BMC/bone_len_max.npy", max_bone_len)

    # calculate root bone limits
    root_bones = kin_chain[:, [0, 4, 8, 12, 16], :]
    normals = normalize(np.cross(root_bones[:, 1:], root_bones[:, :-1]))

    edge_normals = np.zeros_like(root_bones)
    edge_normals[:, 0] = normals[:, 0]
    edge_normals[:, 4] = normals[:, 3]
    edge_normals[:, 1:4] = normalize(normals[:, 1:] + normals[:, :-1])

    curvatures = np.zeros([joints.shape[0], 4])
    PHI = np.zeros([joints.shape[0], 4])
    for i in range(4):
        e_tmp = edge_normals[:, i + 1] - edge_normals[:, i]
        b_tmp = root_bones[:, i + 1] - root_bones[:, i]
        b_tmp_norm = np.linalg.norm(
            b_tmp, ord=2, axis=-1  # N
        )

        curvatures[:, i] = np.sum(e_tmp * b_tmp, axis=-1) / (b_tmp_norm ** 2)
        PHI[:, i] = np.sum(root_bones[:, i] * root_bones[:, i + 1], axis=-1)

        tmp1 = np.linalg.norm(root_bones[:, i], ord=2, axis=-1)
        tmp2 = np.linalg.norm(root_bones[:, i + 1], ord=2, axis=-1)
        PHI[:, i] /= (tmp1 * tmp2)
        PHI[:, i] = np.arccos(PHI[:, i])

    max_curvatures = np.max(curvatures, axis=0)  # (4,)
    min_curvatures = np.min(curvatures, axis=0)  # (4,)
    np.save("BMC/curvatures_max.npy", max_curvatures)
    np.save("BMC/curvatures_min.npy", min_curvatures)

    max_PHI = np.max(PHI, axis=0)  # (4,)
    min_PHI = np.min(PHI, axis=0)  # (4,)
    np.save("BMC/PHI_max.npy", max_PHI)
    np.save("BMC/PHI_min.npy", min_PHI)

    jas = []
    for i in tqdm(range(joints.shape[0])):
        joint = joints[i]
        ja = cal_joint_angles.calculate_joint_angle(joint)
        jas.append(ja)
    jas = np.array(jas)  # (N, 15, 2)
    np.save("BMC/joint_angles.npy", jas)