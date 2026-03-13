import cv2
import torch
from manopth import manolayer
from model.detnet import detnet
from utils import func, bone, AIK, smoother
import numpy as np
import matplotlib.pyplot as plt
from utils import vis
from op_pso import PSO
import open3d
from model import shape_net
import os
import csv
import json
from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_mano_root = 'mano/models'

module = detnet().to(device)
print('load model start')
check_point = torch.load('new_check_point/ckp_detnet_83.pth', map_location=device)
model_state = module.state_dict()
state = {}
for k, v in check_point.items():
    if k in model_state:
        state[k] = v
    else:
        print(k, ' is NOT in current model')
model_state.update(state)
module.load_state_dict(model_state)
print('load model finished')

shape_model = shape_net.ShapeNet()
shape_net.load_checkpoint(
    shape_model, os.path.join('checkpoints', 'ckp_siknet_synth_41.pth.tar')
)
for params in shape_model.parameters():
    params.requires_grad = False

pose, shape = func.initiate("zero")
pre_useful_bone_len = np.zeros((1, 15))
pose0 = torch.eye(3).repeat(1, 16, 1, 1)

mano = manolayer.ManoLayer(flat_hand_mean=True,
                           side="right",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='rotmat',
                           joint_rot_mode='rotmat').to(device)
mano_faces = mano.th_faces
if torch.is_tensor(mano_faces):
    mano_faces = mano_faces.detach().cpu().numpy()
mano_faces = mano_faces.astype(np.int32)

print('start opencv')
point_fliter = smoother.OneEuroFilter(4.0, 0.0)
mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)
shape_fliter = smoother.OneEuroFilter(4.0, 0.0)
cap = cv2.VideoCapture(0)
print('opencv finished')
flag = 1
plt.ion()
f = plt.figure()

fliter_ax = f.add_subplot(111, projection='3d')
plt.show()
view_mat = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0],
                     [0.0, 0, -1.0]])
mesh = open3d.geometry.TriangleMesh()
pose0 = pose0.to(device)
shape = shape.to(device)
hand_verts, j3d_recon = mano(pose0, shape.float())
hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
mesh.triangles = open3d.utility.Vector3iVector(mano_faces)

viewer = open3d.visualization.Visualizer()
viewer.create_window(width=480, height=480, window_name='mesh')
viewer.add_geometry(mesh)
viewer.update_renderer()

print('start pose estimate')

pre_uv = None
shape_time = 0
opt_shape = None
shape_flag = True

# 输出目录
SAVE_DIR = 'saved_hand_predictions'
os.makedirs(SAVE_DIR, exist_ok=True)

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
SAVE_DIR = os.path.join(SAVE_DIR, f'run_{run_timestamp}')
os.makedirs(SAVE_DIR, exist_ok=True)

# 时序缓存
all_frame_ids = []
all_raw_joints = []         # 每帧原始21关节坐标，[21, 3]
all_filtered_joints = []    # 每帧滤波后21关节坐标，[21, 3]
all_bone_lengths = []       # 每帧bone length，[15]
all_shape_beta = []         # 每帧shape beta
all_pose_R = []             # 每帧rotmat姿态
all_j3d_recon = []          # 每帧Mano重建后的21关节坐标
all_uv = []                 # 每帧uv

def save_flatten_csv(file_path, frame_ids, data_array, prefix):
    """
    将任意[T, ...]的数组拉平成二维后保存为 csv
    第一列为 frame_id
    """
    data_array = np.asarray(data_array)
    if data_array.ndim < 2:
        data_array = data_array.reshape(len(frame_ids), 1)

    flat_array = data_array.reshape(data_array.shape[0], -1)

    headers = ['frame_id']
    for i in range(flat_array.shape[1]):
        headers.append(f'{prefix}_{i}')

    with open(file_path, 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(headers)
        for idx, row in zip(frame_ids, flat_array):
            writer.writerow([idx] + row.tolist())


def save_joint_var_metrics(save_dir, raw_joints_array):
    """
    raw_joints_array: [T, 21, 3]
    保存：
    1) 每个关节、每个坐标轴的方差 [21, 3]
    2) 每个关节的整体方差（对3个axis求均值）[21]
    3) 全局整体方差 scalar
    """
    var_per_joint_per_axis = np.var(raw_joints_array, axis=0)   # [21, 3]
    var_per_joint = np.mean(var_per_joint_per_axis, axis=1)     # [21]
    overall_var = np.var(raw_joints_array)

    np.savez(
        os.path.join(save_dir, 'joints_var_metrics.npz'),
        var_per_joint_per_axis=var_per_joint_per_axis,
        var_per_joint=var_per_joint,
        overall_var=overall_var
    )

    # 保存每个关节每个坐标轴的方差
    with open(os.path.join(save_dir, 'joints_var_per_joint_per_axis.csv'), 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['joint_id', 'var_x', 'var_y', 'var_z'])
        for joint_id in range(var_per_joint_per_axis.shape[0]):
            writer.writerow([
                joint_id,
                var_per_joint_per_axis[joint_id, 0],
                var_per_joint_per_axis[joint_id, 1],
                var_per_joint_per_axis[joint_id, 2]
            ])

    # 保存每个关节整体方差
    with open(os.path.join(save_dir, 'joints_var_per_joint.csv'), 'w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['joint_id', 'var_mean_xyz'])
        for joint_id in range(var_per_joint.shape[0]):
            writer.writerow([joint_id, var_per_joint[joint_id]])

    summary = {
        'num_frames': int(raw_joints_array.shape[0]),
        'num_joints': int(raw_joints_array.shape[1]),
        'coord_dim': int(raw_joints_array.shape[2]),
        'overall_var': float(overall_var)
    }
    with open(os.path.join(save_dir, 'joints_var_summary.json'), 'w') as f_json:
        json.dump(summary, f_json, indent=4)

while (cap.isOpened()):
    ret_flag, img = cap.read()
    if not ret_flag:
        break

    input = np.flip(img.copy(), -1)
    if input.shape[0] > input.shape[1]:
        margin = (input.shape[0] - input.shape[1]) // 2
        input = input[margin:-margin]
    else:
        margin = (input.shape[1] - input.shape[0]) // 2
        input = input[:, margin:-margin]

    img = input.copy()
    img = np.flip(img, -1)

    cv2.imshow("Capture_Test", img)
    k = cv2.waitKey(1) & 0xFF
    if k != 255:
      print("key pressed:", k)

    input = cv2.resize(input, (128, 128))
    input = torch.tensor(input.transpose([2, 0, 1]), dtype=torch.float, device=device)  # hwc -> chw
    input = func.normalize(input, [0.5, 0.5, 0.5], [1, 1, 1])
    result = module(input.unsqueeze(0))

    pre_joints = result['xyz'].squeeze(0)
    now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0]
    now_uv = now_uv.astype(np.float)
    trans = np.zeros((1, 3))
    trans[0, 0:2] = now_uv - 16.0
    trans = trans / 16.0
    new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])
    pre_joints = pre_joints.clone().detach().cpu().numpy()

    flited_joints = point_fliter.process(pre_joints)

    fliter_ax.cla()

    filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
    pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

    shape_model_input = torch.tensor(pre_useful_bone_len, dtype=torch.float)
    shape_model_input = shape_model_input.reshape((1, 15))
    dl_shape = shape_model(shape_model_input)
    dl_shape = dl_shape['beta'].numpy()
    dl_shape = shape_fliter.process(dl_shape)
    opt_tensor_shape = torch.tensor(dl_shape, dtype=torch.float)
    opt_tensor_shape = opt_tensor_shape.to(device)
    _, j3d_p0_ops = mano(pose0, opt_tensor_shape)
    template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0  # template, m 21*3
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
    j3d_pre_process = pre_joints * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
    pose_R = AIK.adaptive_IK(template, j3d_pre_process)
    pose_R = torch.from_numpy(pose_R).float()
    pose_R = pose_R.to(device)
    #  reconstruction
    hand_verts, j3d_recon = mano(pose_R, opt_tensor_shape.float())
    mesh.triangles = open3d.utility.Vector3iVector(mano_faces)
    hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
    hand_verts = mesh_fliter.process(hand_verts)
    hand_verts = np.matmul(view_mat, hand_verts.T).T
    hand_verts[:, 0] = hand_verts[:, 0] - 50
    hand_verts[:, 1] = hand_verts[:, 1] - 50
    mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
    hand_verts = hand_verts - 100 * mesh_tran

    # 每帧时序物理信息缓存
    current_frame_id = len(all_frame_ids)

    all_frame_ids.append(current_frame_id)
    all_raw_joints.append(pre_joints.copy())
    all_filtered_joints.append(np.array(flited_joints).copy())
    all_bone_lengths.append(np.array(pre_useful_bone_len).reshape(-1).copy())
    all_shape_beta.append(np.array(dl_shape).reshape(-1).copy())
    all_pose_R.append(pose_R.detach().cpu().numpy().copy())
    all_j3d_recon.append(j3d_recon.detach().cpu().numpy().squeeze(0).copy())
    all_uv.append(np.array(now_uv).reshape(-1).copy())


    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)
    viewer.poll_events()
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(all_frame_ids) > 0:
    raw_joints_array = np.asarray(all_raw_joints)                 # [T, 21, 3]
    filtered_joints_array = np.asarray(all_filtered_joints)       # [T, 21, 3]
    bone_lengths_array = np.asarray(all_bone_lengths)             # [T, 15]
    shape_beta_array = np.asarray(all_shape_beta)                 # [T, beta_dim]
    pose_R_array = np.asarray(all_pose_R)                         # [T, ...]
    j3d_recon_array = np.asarray(all_j3d_recon)                   # [T, 21, 3]
    uv_array = np.asarray(all_uv)                                 # [T, uv_dim]

    np.savez(
        os.path.join(SAVE_DIR, 'hand_physical_info_timeseries.npz'),
        frame_ids=np.asarray(all_frame_ids),
        raw_joints=raw_joints_array,
        filtered_joints=filtered_joints_array,
        bone_lengths=bone_lengths_array,
        shape_beta=shape_beta_array,
        pose_R=pose_R_array,
        j3d_recon=j3d_recon_array,
        uv=uv_array
    )

    save_flatten_csv(
        os.path.join(SAVE_DIR, 'raw_joints_timeseries.csv'),
        all_frame_ids,
        raw_joints_array,
        'raw_joint'
    )
    save_flatten_csv(
        os.path.join(SAVE_DIR, 'filtered_joints_timeseries.csv'),
        all_frame_ids,
        filtered_joints_array,
        'filtered_joint'
    )
    save_flatten_csv(
        os.path.join(SAVE_DIR, 'bone_lengths_timeseries.csv'),
        all_frame_ids,
        bone_lengths_array,
        'bone_len'
    )
    save_flatten_csv(
        os.path.join(SAVE_DIR, 'shape_beta_timeseries.csv'),
        all_frame_ids,
        shape_beta_array,
        'shape_beta'
    )
    save_flatten_csv(
        os.path.join(SAVE_DIR, 'pose_R_timeseries.csv'),
        all_frame_ids,
        pose_R_array,
        'pose_R'
    )
    save_flatten_csv(
        os.path.join(SAVE_DIR, 'j3d_recon_timeseries.csv'),
        all_frame_ids,
        j3d_recon_array,
        'j3d_recon'
    )
    save_flatten_csv(
        os.path.join(SAVE_DIR, 'uv_timeseries.csv'),
        all_frame_ids,
        uv_array,
        'uv'
    )

    save_joint_var_metrics(SAVE_DIR, raw_joints_array)

    print('Saved time-series physical predictions and var metrics to:', SAVE_DIR)
else:
    print('No frame data collected, nothing to save.')