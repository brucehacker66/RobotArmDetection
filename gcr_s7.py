# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:52:49 2021

@author: PQD
"""

# import ctypes
import time
import pyrealsense2 as rs
import numpy as np
import torch
from pred_model import GADGCNN
from model10 import SADGCNN
# from model1 import SADGCNN as SADGCNN2
import open3d as o3d
import rangeimage_op
import utilx
import sys
import assemble_gcr
import socket
import struct
import threading
from rangeimage.rangeimage import depthmap, depthmap2
from DucoCobot import DucoCobot
sys.path.append('gen_py')
sys.path.append('lib')
# from thrift import Thrift
# from thrift.transport import TSocket
# from thrift.transport import TTransport
# from thrift.protocol import TBinaryProtocol
from gen_py.robot.ttypes import StateRobot, StateProgram, OperationMode,TaskState,Op

#%%
control = 0
connect = 0
if control:
    utilx.init_logger()
    cli = utilx.connectModbus()

def initRobot():
    robotIP = '192.168.1.10'
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (robotIP, 2001)
    client_socket.connect(server_address)
    return client_socket

global duco_cobot
if connect:
    duco_cobot = initRobot()

#%%
def dis_mink(a, b, k):
    inner = -2 * torch.matmul(a, b.t())
    a2 = torch.sum(a**2, dim=1, keepdim=True)
    b2 = torch.sum(b**2, dim=1, keepdim=True)
    dis_mat = -a2 - inner - b2.t()
    dis_mat = dis_mat.max(dim=1)[0]
    dis = dis_mat.topk(k=k, dim=-1)[0]
    return -dis

#%%
fov = 1.8
half_filter = 1
thres = 0.05

width = 640
height = 480
fps = 30
p = rs.pipeline()
config = rs.config()
# config.enable_device_from_file("000.bag")
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

p.start(config)
align_to_color = rs.align(rs.stream.color)
pc = rs.pointcloud()
points = rs.points()
#decimation_filter = rs.decimation_filter()
threshold_filter = rs.threshold_filter(0.15, 10)
spatial_filter = rs.spatial_filter(0.5, 20.0, 2.0, 0.0)
temp_filter = rs.temporal_filter(0.4, 20, 3)

#mat = np.array([[0.067503, -0.768993, 0.635683, -2.375150],
#                [-0.996426, -0.019529, 0.082186, -0.339808],
#                [-0.050786, -0.638959, -0.767563, 2.772721],
#                [0.000000, 0.000000, 0.000000, 1.000000]])
# mat = np.array([[0.998225, -0.036355, -0.047167, 0.332170],
#                 [0.000000, -0.792039, 0.610470, -1.809778],
#                 [-0.059552, -0.609387, -0.790633, 2.799637],
#                 [0.000000, 0.000000, 0.000000, 1.000000]])
# mat = np.array([[0.998492, 0.028307, 0.047035, 0.194272],
#                 [0.000000, -0.856800, 0.515648, -1.333229],
#                 [0.054896, -0.514871, -0.855508, 2.877786],
#                 [0.000000, 0.000000, 0.000000, 1.000000]])
mat = np.array([[0.995272, -0.043597, 0.086789, 0.953730],
                [-0.094156, -0.652311, 0.752080, -2.827866],
                [0.023825, -0.756696, -0.653332, 3.200157],
                [0.000000, 0.000000, 0.000000, 1.000000]])
R = mat[:3, :3].transpose()
T = mat[:3, 3]
R_gpu = torch.from_numpy(R.transpose()).float().cuda()
T_gpu = torch.from_numpy(T).float().cuda()
T_gpu = torch.matmul(-T_gpu, R_gpu)

def get_point():
    
    frames = p.wait_for_frames()
    frames = align_to_color.process(frames)
    depth = frames.get_depth_frame()
    color = frames.get_color_frame()
    
#    depth = decimation_filter.process(depth)
    depth = threshold_filter.process(depth)
    depth = spatial_filter.process(depth)
    depth = temp_filter.process(depth)
    pc.map_to(color)
    points = pc.calculate(depth)
    vtx = np.array(points.get_vertices())
    vtx = vtx.view('f4')
    vtx = vtx.reshape([-1, 3])
    
    colorful = np.array(color.get_data()) / 255.0
    colorful = colorful.reshape(-1,3)
    vtx = np.concatenate([vtx, colorful], axis=-1)
    mask = np.nonzero(vtx[:, 0])
    vtx = vtx[mask, :]
    vtx = vtx.reshape(-1, 6)
    return vtx

def filters(points):
    pic_data = points * torch.tensor([width/fov, width/fov, 1.], dtype=torch.float, device=points.device)
    pic_data[:, 0] = pic_data[:, 0] / pic_data[:, 2]
    pic_data[:, 1] = pic_data[:, 1] / pic_data[:, 2]
    t2c_vector = torch.tensor([width/2, height/2, 0.], dtype=torch.float, device=points.device)
    pic_data = pic_data + t2c_vector
    
    image = rangeimage_op.trans_depth(pic_data, height, width, 3)
    image = image.min(dim=2)[0]
    
    image = rangeimage_op.depth_filter(image, 0.1)
    pc = rangeimage_op.trans_point(image, half_filter, thres)
    mask = np.nonzero(pc[:, 0].cpu().numpy())
    pc = pc[mask, :]
    pc = pc.squeeze(0)
    
    pc = pc - t2c_vector
    pc[:, 0] = pc[:, 0] * pc[:, 2]
    pc[:, 1] = pc[:, 1] * pc[:, 2]
    pc = pc * torch.tensor([fov/width, fov/width, 1.], dtype=torch.float, device=pc.device)
    return pc
#%%
device = torch.device('cuda')
seg = SADGCNN(3, k1=16, bn=True).to(device)
seg.load_state_dict(torch.load('model10s.pt'))
seg.eval()

seg2 = SADGCNN(3, k1=16, bn=True).to(device)
seg2.load_state_dict(torch.load('model10.pt'))
seg2.eval()

#cam_in_base = np.loadtxt('mat.txt')
#R = cam_in_base[:3, :3].transpose()
#T = cam_in_base[:3, 3]

color_select = np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0]])

#%%
now_time = time.time()
# vtx = get_point()
# pc1 = o3d.geometry.PointCloud()
# pc1.points = o3d.utility.Vector3dVector(vtx[:, :3])
# pc1.colors = o3d.utility.Vector3dVector(vtx[:, 3:])
# pc2 = o3d.geometry.PointCloud()
vtx = get_point()
permutation = np.random.permutation(vtx.shape[0])
permutation = permutation[:2048]
vtx = vtx[permutation, :]
point = np.dot(vtx[:, :3], R)
point = point + T
# point = vtx[:, :3] - T
# point = np.dot(point, R)
# point = point + [-0.61, 0.0, 0.0]

# pc2.points = o3d.utility.Vector3dVector(point)

# vis = o3d.visualization.Visualizer()
# vis.create_window(width=800, height=600)
# vis.add_geometry(pc1)
#
# vis2 = o3d.visualization.Visualizer()
# vis2.create_window(width=800, height=600, left=1000, top=50)
# vis2.add_geometry(pc2)

# sts = ctypes.c_int(1)
# psts = ctypes.pointer(sts)

scene2 = np.loadtxt('scene_5.txt')
#permutation = np.random.permutation(scene2.shape[0])
#permutation = permutation[:2048]
#scene2 = scene2[permutation, :]
#np.savetxt('scene_.txt', scene2, fmt='%.6f')
scene2 = torch.from_numpy(scene2).float().cuda()
scene2 = scene2.view(1, -1, 3).contiguous()
scene2_ = scene2.view(-1, 3).contiguous()

pose = []

def get_pose():
    global pose
    global duco_cobot
    while(1):
        try:
            pose_ = []
            pose_data = duco_cobot.recv(1468)
            for j in range(6):
                pose_.append(struct.unpack('f', pose_data[j * 4:j * 4 + 4]))
            pose_ = np.array(pose_).reshape(-1)
            pose = pose_
            time.sleep(0.01)
        except:
            duco_cobot = initRobot()
            pose = np.array([0., 0., 0., 0., 0., 0.])
            print('Reboot robot.')

if __name__ == '__main__':
    if connect:
        t1 = threading.Thread(target=get_pose)
        t1.setDaemon(True)
        t1.start()
    
    f = 0
    while(1):
        last_time = now_time
        now_time = time.time()
        print(now_time - last_time)
        vtx = get_point()
        
        # pc1.points = o3d.utility.Vector3dVector(vtx[:, :3])
        # pc1.colors = o3d.utility.Vector3dVector(vtx[:, 3:])
        # vis.update_geometry(pc1)
        # vis.poll_events()
        # vis.update_renderer()
        # # o3d.io.write_point_cloud('video13/1.pcd', pc1)
        
        vtx_g = torch.from_numpy(vtx[:, :3]).float().cuda()
        vtx = filters(vtx_g).cpu().numpy()
        # np.savetxt('video13/1.txt', vtx, fmt='%.6f')
        
        point = np.dot(vtx, R)
        point = point + T
        # point = vtx[:, :3] - T
        # point = np.dot(point, R)
        # point = point + [-0.61, 0.0, 0.0]
        mask = point[:, 2] > 0.15
        point = point[mask, :]
        point = point.reshape(-1, 3)
        mask = point[:, 2] < 2.4
        point = point[mask, :]
        point = point.reshape(-1, 3)
        mask = point[:, 1] < 2.1
        point = point[mask, :]
        point = point.reshape(-1, 3)
        mask = point[:, 0] > -1.3
        point = point[mask, :]
        point = point.reshape(-1, 3)
        mask = point[:, 0] < 2.6
        point = point[mask, :]
        point = point.reshape(-1, 3)

        mask1 = point[:, 0] > 0.5
        mask2 = point[:, 0] < 1.6
        mask3 = point[:, 1] > -0.6
        mask4 = point[:, 1] < 0.4
        mask5 = point[:, 2] < 1.0
        mask = np.logical_and(mask1, mask2)
        mask = np.logical_and(mask, mask3)
        mask = np.logical_and(mask, mask4)
        mask = np.logical_and(mask, mask5)
        mask1 = point[:, 0] > 1.4
        mask2 = point[:, 1] > 1.15
        mask_ = np.logical_and(mask1, mask2)
        mask = np.logical_or(mask, mask_)
        mask = np.logical_not(mask)
        point = point[mask, :]
        point = point.reshape(-1, 3)
        # np.savetxt('point.txt', point, fmt='%.6f')
        permutation = np.random.permutation(point.shape[0])
        permutation = permutation[:2048]
        point = point[permutation, :]
        # pc2.points = o3d.utility.Vector3dVector(point)
        data = torch.from_numpy(point).float().cuda()
        indata = data.view(1, -1, 3).contiguous()
        
        if connect:
            # pose = []
            # try:
            #     pose_data = duco_cobot.recv(1468)
            #     for i in range(6):
            #         pose.append(struct.unpack('f', pose_data[i * 4:i * 4 + 4]))
            #     pose = np.array(pose).reshape(-1)
            # except:
            #     duco_cobot = initRobot()
            #     pose = np.array([0., 0., 0., 0., 0., 0.])
            #     print('Reboot robot.')
            print(pose)
            if pose.any() == 0:
                pred = seg2(indata, scene2)
            else:
                pose = np.rad2deg(pose)
                pose += [-90., 0., -90., 0., 90., -45.]
                robot = assemble_gcr.robot_pose(pose)
                robot = torch.from_numpy(robot).float().cuda()
                robot = depthmap(robot, R_gpu, T_gpu, 640, 480, 2.4, 2, 0.05)
                robot = torch.cat([scene2_, robot], dim=0)
                permutation = np.random.permutation(robot.shape[0])
                permutation = permutation[:2048]
                robot = robot[permutation, :]
                # np.savetxt('robot1.txt', robot.cpu().numpy(), fmt='%.6f')
                robot = robot.view(1, -1, 3).contiguous()
                pred = seg(indata, robot)
        else:
            pred = seg2(indata, scene2)
        
        pred = pred.squeeze(0).t()
        pred = pred.max(dim=1)[1].int()
        # label = pred.cpu().numpy()
        # color = color_select[label, :]
        # pc2.colors = o3d.utility.Vector3dVector(color)
        # vis2.update_geometry(pc2)
        # vis2.poll_events()
        # vis2.update_renderer()
        
        k = 25
        is_close = 0
        dis_thres = 1.2 * 1.2
        obstacle_select = pred > 1
        has_obs = obstacle_select.count_nonzero()
        
        if has_obs > k:
            obstacle_select = torch.nonzero(obstacle_select)
            obstacle = data[obstacle_select, :].squeeze(1)
            robot_select = pred == 1
            if robot_select.count_nonzero() > k:
                robot_select = torch.nonzero(robot_select)
                robot = data[robot_select, :]
                robot = robot.squeeze(1)
                
                dis = dis_mink(obstacle, robot, k).mean()
                if dis < dis_thres:
                    is_close = 1
        if is_close == 1:
            f = 1
            if control:
                utilx.sendDataToModbus2(2,1)
        if f:
            f += 1
        if is_close == 0 and f > 20:
            f = 0
            if control:
                utilx.sendDataToModbus2(2,0)
        print(f)

# vis.destroy_window()
# vis2.destroy_window()
p.stop()
# endIno()