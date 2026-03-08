import cv2
import torch
from manopth import manolayer
from model.detnet import detnet
from utils import func, bone, AIK, smoother
import numpy as np
import open3d
from model import shape_net
import os

# ================= 配置区域 =================
# [关键修改] 这里强制写成 0，表示使用默认摄像头
CAMERA_INDEX = 0  
# 权重文件路径 (请确保文件存在)
PATH_DETNET = 'new_check_point/ckp_detnet_83.pth'
PATH_SHAPENET = 'checkpoints/ckp_siknet_synth_41.pth.tar'
# ===========================================

# 1. 设备配置 (适配 RTX 4070Ti)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"当前运行设备: {torch.cuda.get_device_name(0)}")

_mano_root = 'mano/models'

# 2. 加载 DetNet (手势模型)
print('正在加载 DetNet...')
module = detnet().to(device)
if not os.path.exists(PATH_DETNET):
    raise FileNotFoundError(f"找不到文件: {PATH_DETNET}，请确认路径！")
check_point = torch.load(PATH_DETNET, map_location=device)
module.load_state_dict(check_point, strict=False)

# 3. 加载 ShapeNet (手型模型)
print('正在加载 ShapeNet...')
shape_model = shape_net.ShapeNet().to(device)
if os.path.exists(PATH_SHAPENET):
    # 如果找不到 map_location 参数，就去掉它
    try:
        shape_net.load_checkpoint(shape_model, PATH_SHAPENET, map_location=device)
    except TypeError:
        shape_net.load_checkpoint(shape_model, PATH_SHAPENET)
else:
    print(f"[警告] 找不到 {PATH_SHAPENET}，手型将随机生成（但这不影响动作）！")

for params in shape_model.parameters():
    params.requires_grad = False

# 4. 初始化 MANO 和变量 (全部搬到 GPU)
pose, shape = func.initiate("zero")
pose0 = torch.eye(3).repeat(1, 16, 1, 1).to(device)
shape = shape.to(device)
pre_useful_bone_len = np.zeros((1, 15))

mano = manolayer.ManoLayer(flat_hand_mean=True,
                           side="right",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='rotmat',
                           joint_rot_mode='rotmat').to(device)

# 5. 初始化滤波器 (防抖动)
point_fliter = smoother.OneEuroFilter(1.5, 0.0)
mesh_fliter = smoother.OneEuroFilter(1.5, 0.0)
shape_fliter = smoother.OneEuroFilter(1.5, 0.0)

# 6. 初始化摄像头
print(f'正在尝试打开摄像头 (ID: {CAMERA_INDEX})...')
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    print("尝试方法：1. 检查隐私权限；2. 拔插摄像头；3. 将 CAMERA_INDEX 改为 1")
    raise RuntimeError("摄像头启动失败")

# 7. 初始化 Open3D 可视化窗口
print("正在初始化 3D 窗口...")
view_mat = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0], [0.0, 0, -1.0]])
mesh = open3d.geometry.TriangleMesh()

# [关键修复] 预热 MANO 并修复 Open3D 类型报错
hand_verts, _ = mano(pose0, shape.float())
# GPU tensor -> CPU numpy -> int32 (解决 Vector3iVector 报错)
mano_faces = mano.th_faces.detach().cpu().numpy().astype(np.int32)
mesh.triangles = open3d.utility.Vector3iVector(mano_faces)
# GPU tensor -> CPU numpy -> float64 (解决 vertices 报错)
hand_verts = hand_verts.clone().detach().cpu().numpy()[0].astype(np.float64)
mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.9, 0.7, 0.58]) # 肤色

vis = open3d.visualization.Visualizer()
vis.create_window(width=640, height=480, window_name='3D Mesh Result (Right Hand)', left=800, top=100)
vis.add_geometry(mesh)

print('>>> 系统就绪！左边看摄像头，右边看3D手势。按 "q" 键退出 <<<')

while cap.isOpened():
    ret, img = cap.read()
    if not ret: 
        print("无法读取视频流")
        break

    # === 图像预处理 ===
    # 裁剪正方形
    h, w, _ = img.shape
    if h > w:
        margin = (h - w) // 2
        input_img = img[margin:-margin, :]
    else:
        margin = (w - h) // 2
        input_img = img[:, margin:-margin]
    
    # 镜像翻转，像照镜子一样自然
    #input_img = np.flip(input_img, 1)
    
    # 准备模型输入
    input_tensor = cv2.resize(input_img, (128, 128))
    # 这里的 .cuda() 或 .to(device) 很重要
    input_tensor = torch.tensor(input_tensor.transpose([2, 0, 1]), dtype=torch.float).to(device)
    input_tensor = func.normalize(input_tensor, [0.5, 0.5, 0.5], [1, 1, 1])

    # === 模型推理 ===
    result = module(input_tensor.unsqueeze(0))
    
    # 解析数据
    pre_joints = result['xyz'].squeeze(0)
    now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0].astype(float)
    
    # 坐标校准
    trans = np.zeros((1, 3))
    trans[0, 0:2] = now_uv - 16.0
    trans = trans / 16.0
    new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])

    # 滤波
    pre_joints = pre_joints.clone().detach().cpu().numpy()
    flited_joints = point_fliter.process(pre_joints)
    pre_useful_bone_len = bone.caculate_length(flited_joints, label="useful")

    # 手型估计 (ShapeNet)
    shape_input = torch.tensor(pre_useful_bone_len, dtype=torch.float).to(device).reshape((1, 15))
    dl_shape = shape_model(shape_input)['beta'].detach().cpu().numpy()
    dl_shape = shape_fliter.process(dl_shape)
    opt_tensor_shape = torch.tensor(dl_shape, dtype=torch.float).to(device)

    # === 3D 重建 ===
    # 1. 计算 Template
    _, j3d_p0_ops = mano(pose0, opt_tensor_shape)
    template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0
    
    # 2. 缩放匹配
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(flited_joints[9] - flited_joints[0])
    j3d_pre_process = flited_joints * ratio
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]

    # 3. IK (反向运动学)
    pose_R = AIK.adaptive_IK(template, j3d_pre_process)
    pose_R = torch.from_numpy(pose_R).float().to(device)

    # 4. 生成最终网格
    hand_verts, _ = mano(pose_R, opt_tensor_shape.float())
    
    # === 可视化更新 ===
    
    # 1. 更新 3D 窗口
    hand_verts = hand_verts.clone().detach().cpu().numpy()[0].astype(np.float64)
    hand_verts = mesh_fliter.process(hand_verts)
    
    # 对齐视角
    hand_verts = np.matmul(view_mat, hand_verts.T).T
    hand_verts[:, 0] -= 50
    hand_verts[:, 1] -= 50
    mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
    hand_verts -= 100 * mesh_tran
    
    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    mesh.compute_vertex_normals()
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # 2. 更新 OpenCV 摄像头窗口 (画骨架)
    display_img = input_img.copy()
    joints_2d = pre_joints[:, :2] # 取 X, Y
    
    # 骨架连线索引
    bones_idx = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (0,9),(9,10),(10,11),(11,12), 
                 (0,13),(13,14),(14,15),(15,16), (0,17),(17,18),(18,19),(19,20)]
    
    h_disp, w_disp = display_img.shape[0], display_img.shape[1]
    
    # 画红线绿点
    for start, end in bones_idx:
        # 坐标映射 -1~1 -> 图像像素
        p1 = (int((joints_2d[start, 0] + 1) / 2 * w_disp), int((joints_2d[start, 1] + 1) / 2 * h_disp))
        p2 = (int((joints_2d[end, 0] + 1) / 2 * w_disp), int((joints_2d[end, 1] + 1) / 2 * h_disp))
        cv2.line(display_img, p1, p2, (0, 0, 255), 2)
        cv2.circle(display_img, p1, 4, (0, 255, 0), -1)

    cv2.imshow("Camera Input (Press 'q' to exit)", input_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vis.destroy_window()