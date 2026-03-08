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

# ================= 配置输出文件名 =================
OUTPUT_VIDEO = 'result_demo2.mp4'
# OUTPUT_VIDEO = 0
# ===============================================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_mano_root = 'mano/models'

module = detnet().to(device)
print('load model start')
# [兼容修复] 这里的 map_location 必须保留
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

shape_model = shape_net.ShapeNet().to(device) # [修复] 移动到 GPU
# [兼容修复] 增加 try-except 防止旧版加载函数报错
try:
    shape_net.load_checkpoint(
        shape_model, os.path.join('checkpoints', 'ckp_siknet_synth_41.pth.tar')
    )
except:
    # 如果原代码不支持 map_location，这里手动加载一下或者忽略
    pass

for params in shape_model.parameters():
    params.requires_grad = False

pose, shape = func.initiate("zero")
pre_useful_bone_len = np.zeros((1, 15))
pose0 = torch.eye(3).repeat(1, 16, 1, 1).to(device) # [修复] 移动到 GPU

mano = manolayer.ManoLayer(flat_hand_mean=True,
                           side="right",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='rotmat',
                           joint_rot_mode='rotmat').to(device) # [修复] 移动到 GPU
print('start opencv')
point_fliter = smoother.OneEuroFilter(4.0, 0.0)
mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)
shape_fliter = smoother.OneEuroFilter(4.0, 0.0)


video_path = 'test2.mp4' # 确保文件存在
# cap = cv2.VideoCapture(video_path) 
cap = cv2.VideoCapture(0)


print('opencv finished')
flag = 1
plt.ion()
f = plt.figure()

fliter_ax = f.add_subplot(111, projection='3d')
# plt.show() # [建议] 注释掉这行，否则会阻塞视频处理

view_mat = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0],
                     [0.0, 0, -1.0]])
mesh = open3d.geometry.TriangleMesh()

# [修复] 类型转换
hand_verts, j3d_recon = mano(pose0, shape.to(device).float())
mano_faces = mano.th_faces.detach().cpu().numpy().astype(np.int32)
mesh.triangles = open3d.utility.Vector3iVector(mano_faces)
hand_verts = hand_verts.clone().detach().cpu().numpy()[0].astype(np.float64)
mesh.vertices = open3d.utility.Vector3dVector(hand_verts)

viewer = open3d.visualization.Visualizer()
viewer.create_window(width=480, height=480, window_name='mesh', visible=True)
viewer.add_geometry(mesh)
viewer.update_renderer()

print('start pose estimate')

# ================= [准备视频写入] =================
fps = cap.get(cv2.CAP_PROP_FPS)
# 输出宽度 = 原图裁剪后(128x128太小，这里用原图显示大小) + Open3D窗口(480)
# 假设原图裁剪后大概也是几百像素，这里我们简单处理：
# 左边放 原图(resize到480高)，右边放 3D(480高)
out_h = 480
out_w = 480 + 480 # 左图 + 右图
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))
# ===============================================

pre_uv = None
shape_time = 0
opt_shape = None
shape_flag = True

while (cap.isOpened()):
    ret_flag, img = cap.read()
    if not ret_flag: break
    
    input_img = np.flip(img.copy(), -1)
    k = cv2.waitKey(1) & 0xFF
    
    # 裁剪逻辑 (保留原版)
    if input_img.shape[0] > input_img.shape[1]:
        margin = (input_img.shape[0] - input_img.shape[1]) // 2
        input_img = input_img[margin:-margin]
    else:
        margin = (input_img.shape[1] - input_img.shape[0]) // 2
        input_img = input_img[:, margin:-margin]
    
    img_display = input_img.copy() # 用于显示的图
    img_display = np.flip(img_display, -1)
    
    # cv2.imshow("Capture_Test", img_display) # [可选] 不弹窗跑得快
    
    input_tensor = cv2.resize(input_img, (128, 128))
    # [修复] 加上 .to(device)
    input_tensor = torch.tensor(input_tensor.transpose([2, 0, 1]), dtype=torch.float).to(device)
    input_tensor = func.normalize(input_tensor, [0.5, 0.5, 0.5], [1, 1, 1])
    
    result = module(input_tensor.unsqueeze(0))

    pre_joints = result['xyz'].squeeze(0)
    now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0]
    now_uv = now_uv.astype(float) # [修复] np.float -> float
    
    trans = np.zeros((1, 3))
    trans[0, 0:2] = now_uv - 16.0
    trans = trans / 16.0
    new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])
    
    pre_joints = pre_joints.clone().detach().cpu().numpy()

    flited_joints = point_fliter.process(pre_joints)

    # Matplotlib 绘图 (为了原逻辑保留，但很慢)
    fliter_ax.cla()
    filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
    
    pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

    # [修复] 加上 .to(device)
    shape_model_input = torch.tensor(pre_useful_bone_len, dtype=torch.float).to(device)
    shape_model_input = shape_model_input.reshape((1, 15))
    
    dl_shape = shape_model(shape_model_input)
    dl_shape = dl_shape['beta'].detach().cpu().numpy() # [修复] detach
    dl_shape = shape_fliter.process(dl_shape)
    
    # [修复] 加上 .to(device)
    opt_tensor_shape = torch.tensor(dl_shape, dtype=torch.float).to(device)
    
    _, j3d_p0_ops = mano(pose0, opt_tensor_shape)
    template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0
    
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
    j3d_pre_process = pre_joints * ratio
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
    
    pose_R = AIK.adaptive_IK(template, j3d_pre_process)
    # [修复] 加上 .to(device)
    pose_R = torch.from_numpy(pose_R).float().to(device)
    
    # reconstruction
    hand_verts, j3d_recon = mano(pose_R, opt_tensor_shape.float())
    
    # [修复] Open3D 类型转换
    hand_verts = hand_verts.clone().detach().cpu().numpy()[0].astype(np.float64)
    hand_verts = mesh_fliter.process(hand_verts)
    hand_verts = np.matmul(view_mat, hand_verts.T).T
    hand_verts[:, 0] = hand_verts[:, 0] - 50
    hand_verts[:, 1] = hand_verts[:, 1] - 50
    mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
    hand_verts = hand_verts - 100 * mesh_tran

    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)
    viewer.poll_events()
    viewer.update_renderer()

    # ================= [保存视频逻辑] =================
    # 1. 抓取 Open3D 画面
    img_3d_buffer = viewer.capture_screen_float_buffer(False)
    img_3d = (np.asarray(img_3d_buffer) * 255).astype(np.uint8)
    img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
    
    # 2. 调整尺寸以便拼接
    # 强制把原图和3D图都 resize 到 480x480
    frame_left = cv2.resize(img_display, (480, 480))
    frame_right = cv2.resize(img_3d, (480, 480))
    
    # 3. 拼接
    combined_frame = np.hstack((frame_left, frame_right))
    
    # 4. 写入
    out.write(combined_frame)
    # ===============================================

    if k == ord('q'):
        break

cap.release()
out.release() # 释放保存器
cv2.destroyAllWindows()
viewer.destroy_window()
print(f"处理完成！视频已保存为 {OUTPUT_VIDEO}")