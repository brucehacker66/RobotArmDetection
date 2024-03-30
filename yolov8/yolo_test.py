import cv2
import torch
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import struct
import threading
import time
import open3d as o3d

pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(width=800, height=600)

model = torch.hub.load('/home/wyz/test_code/yolov5-master', 'custom', '/home/wyz/test_code/yolov5-master/yolov5s.pt', source='local')

height = 480
width = 640 
p = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 15)     
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
# config.enable_device_from_file("000.bag")
p.start(config)
align_to_color = rs.align(rs.stream.color)

pc = rs.pointcloud()        # 声明点云对象
points = rs.points()


def get_frame():
    frames = p.wait_for_frames()
    align_frames = align_to_color.process(frames)
    depth = align_frames.get_depth_frame()
    color = frames.get_color_frame()
    pc.map_to(color)
    points = pc.calculate(depth)
    colorpic = np.array(color.get_data())
    pointcloud = np.array(points.get_vertices()).view('f4').reshape(-1, 3)

    colorful = np.array(color.get_data()) / 255.0
    colorful = colorful.reshape(-1,3)
    pointcloud = np.concatenate([pointcloud, colorful], axis=-1).reshape(-1, 6)
    # 增加色彩

    pointcloud = torch.tensor(pointcloud)
    # pointcloud = pointcloud[::25, :] # 每隔25行采样一个点
    # pointcloud = pointcloud[torch.norm(pointcloud, dim=1) != 0] # 去除全0的无效点 
    # pointcloud = pointcloud[(pointcloud[:, 2] != 0) & (pointcloud[:, 2] <= 8)].to('cuda:0') # 去除距离过远的点
    return colorpic, pointcloud

def select(pointcloud, results):
    map = pointcloud.reshape(height, width, -1) # 形成一个和像素对应的点云坐标图，第一维是高，第二维是宽，第三维是点的坐标（即每个像素位置对应一个点云的xyz，有颜色则多三个RGB）
    count = ((results.size())[0]) # 对当前的人的数量进行计数
    samplepoint = 800 # 最终输出的每个人的采样点数
    allpeople = torch.zeros(count * samplepoint, 6) # 预先生成储存若干个人的张量
    i = 0 # 计数
    if count !=0 :
        for det in results:
            xmin, ymin, xmax, ymax, confidence, class_pred = det
            hx = xmax - xmin ; hy = ymax - ymin # 框的长与宽
            scale = 0.06 # 框的内缩比例/2
            xmin, ymin, xmax, ymax = int(xmin + scale * hx), int(ymin + scale * hy), int(xmax - scale * hx), int(ymax - scale * hy)
            people = map[ymin:ymax, xmin:xmax, :] # 筛选出识别框内的所有点云
            people = people.reshape(-1,6)  # 将提取的点云展平为二维
            # people = people[(people[:, 2] != 0) & (people[:, 2] <= 6)] # 去除距离异常的点

            people = people[::25, :] #对点进行初步的等距采样
            pdis = people[:, 2]
            median_dis = torch.median(pdis) # 计算中位数

            people = people[abs(people[:, 2] - median_dis) <= 0.5] # 滤除到中位数距离过远的点
            pdis = people[:, 2] # 再次提取距离
            median_dis = torch.median(pdis) # 再次计算中位数

            pnum = ((people.size())[0]) # 计算点的总数
            dis_to_median = torch.abs(pdis - median_dis) # 到中位数的距离
            index_p = torch.topk(dis_to_median, k=min(samplepoint, pnum), largest=False).indices # 找出最靠近中位数的若干个点的索引
            people = people[index_p]
            if people.numel() == 0: # 防止采样后框过小无点云导致的空张量
                people = torch.zeros(samplepoint, 6)
                print(people)

            if pnum < samplepoint:
                sup = torch.zeros(samplepoint - pnum, 6)
                #sup = people[0].repeat(samplepoint - pnum, 1)
                people = torch.cat((people, sup), dim=0) # 如果原本的点太少，强行补全(不然维度对不齐)

            allpeople[i * samplepoint:(i + 1) * samplepoint] = people[0 : samplepoint]
            i = i + 1
    else:
        allpeople = torch.zeros(1 , 6) # 没人就返回一个1*6的零张量，方便后面计算
    np.savetxt("ap.txt", allpeople.numpy()) 
    print("ok")
    return allpeople
    




def show_frame(image, results):
    for det in results: # 遍历每一行
        xmin, ymin, xmax, ymax, confidence, class_pred = det
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # class_pred = int(class_pred)
        # 在图像上绘制边框和类别
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f'Confidence: {confidence:.2f}', (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # 显示图像
    cv2.imshow('View', image)
    cv2.waitKey(1)  # 等待1毫秒，保证图像窗口能够及时更新

def judge(allpeople):
    limit = 2 # 单位m
    allpeople = allpeople[torch.norm(allpeople, dim=1) != 0] # 去除全0的无效点
    under_limit = allpeople[allpeople[:, 2] < limit]
    warn_points = (under_limit.size())[0]
    print(warn_points)
    if warn_points > 100:
        print("warn!")


def pointview(getpoints):
    vis.reset_view_point(True)
    pointsview = getpoints[:, :3].numpy()
    pointsview[:, 1] = -pointsview[:, 1]
    colorsview = getpoints[:, 3:].numpy()
    pcd.points = o3d.utility.Vector3dVector(pointsview)
    pcd.colors = o3d.utility.Vector3dVector(colorsview)
    # 清空可视化窗口并添加更新后的点云
    vis.clear_geometries()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 12
    # 更新可视化窗口
    vis.poll_events()
    vis.update_renderer()



if __name__ == '__main__':
    while(1):
        a = time.time()
        color, pointcloud = get_frame()
        img = Image.fromarray(color)
        results = model(img)
        all_results = results.xyxy[0]
        people_results = all_results[all_results[:, 5] == 0] # 筛选出人的点
        # print(people_results)
        allpeople = select(pointcloud, people_results)
        # A = np.loadtxt("A.txt")
        # R = torch.tensor(A[:3, :3]).float()
        # o = torch.tensor(A[:3, 3]).float()
        # allpeople[:, :3] = torch.matmul(allpeople[:, :3], R.t()) + o
        # print("distance", allpeople[400, 1])
        judge(allpeople)
        show_frame(color, people_results)
        pointview(allpeople)
        b = time.time()
        print("deal:", b-a)
        # np.savetxt("0.txt", pointcloud.numpy())
        # print("MAP OK")
        # break
            