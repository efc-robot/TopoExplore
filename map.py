#!/usr/bin/python3.8
from tkinter.constants import Y
import rospy
from rospy.rostime import Duration
from rospy.timer import Rate, sleep
from sensor_msgs.msg import Image
import rospkg
import tf
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from torch import jit
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Twist, PoseStamped, Point
from laser_geometry import LaserProjection
import message_filters
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatusArray
from gazebo_msgs.msg import ModelStates

from ros_topoexplore.msg import UnexploredDirectionsMsg
from ros_topoexplore.msg import TopoMapMsg

from ros_topoexplore.srv import ExpStart

from TopoMap import Vertex, Edge, TopologicalMap
from utils.imageretrieval.imageretrievalnet import init_network
from utils.imageretrieval.extract_feature import cal_feature
from utils.topomap_bridge import TopomapToMessage, MessageToTopomap

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

import os
import cv2
from cv_bridge import CvBridge
import numpy as np
from queue import Queue
from scipy.spatial.transform import Rotation as R
import math
import time
import copy

# from pympler.asizeof import asizeof

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
    # new networks with whitening learned end-to-end
    'rSfM120k-tl-resnet50-gem-w'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w'            : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w'           : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

def get_net_param(state):
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', False)
    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    return net_params

def set_marker(robot_name, id, pose, color=(0.5, 0, 0.5), action=Marker.ADD):
    now = rospy.Time.now()
    marker_message = Marker()
    marker_message.header.frame_id = robot_name + "/odom"
    marker_message.header.stamp = now
    marker_message.ns = "topological_map"
    marker_message.id = id
    marker_message.type = Marker.SPHERE
    marker_message.action = action
    marker_message.pose.position = pose.pose.position
    marker_message.pose.orientation = pose.pose.orientation
    marker_message.scale.x = 0.3
    marker_message.scale.y = 0.3
    marker_message.scale.z = 0.3
    marker_message.color.a = 1.0
    marker_message.color.r = color[0]
    marker_message.color.g = color[1]
    marker_message.color.b = color[2]

    return marker_message

def set_edge(robot_name, id, poses, type="edge"):
    now = rospy.Time.now()
    path_message = Marker()
    path_message.header.frame_id = robot_name + "/odom"
    path_message.header.stamp = now
    path_message.ns = "topological_map"
    path_message.id = id
    if type=="edge":
        path_message.type = Marker.LINE_STRIP
        path_message.color.a = 1.0
        path_message.color.r = 0.0
        path_message.color.g = 1.0
        path_message.color.b = 0.0
    elif type=="arrow":
        path_message.type = Marker.ARROW
        path_message.color.a = 1.0
        path_message.color.r = 1.0
        path_message.color.g = 0.0
        path_message.color.b = 0.0
    path_message.action = Marker.ADD
    path_message.scale.x = 0.05
    path_message.scale.y = 0.05
    path_message.scale.z = 0.05
    path_message.points.append(poses[0])
    path_message.points.append(poses[1])

    path_message.pose.orientation.x=0.0
    path_message.pose.orientation.y=0.0
    path_message.pose.orientation.z=0.0
    path_message.pose.orientation.w=1.0

    return path_message


class MapNode:
    def __init__(self, robot_name, robot_list):
        rospack = rospkg.RosPack()
        path = rospack.get_path('ros_topoexplore')

        self.cv_bridge = CvBridge()

        self.map = TopologicalMap(robot_name=robot_name, threshold=0.97)
        self.vertex_dict = dict()
        self.edge_dict = dict()
        self.relative_position = dict()
        self.relative_orientation = dict()
        self.meeted_robot = list()
        for item in robot_list:
            self.vertex_dict[item] = list()
            self.edge_dict[item] = list()
            self.relative_position[item] = [0, 0, 0]
            self.relative_orientation[item] = 0

        self.pose = None
        self.panoramic_view = None
        self.grid_map  = None
        self.global_map = None
        self.global_map_info = None
        self.map_orientation = None
        self.map_angle = None
        self.current_loc = [0,0]

        self.marker_pub = rospy.Publisher(
            robot_name+"/visualization/marker", MarkerArray, queue_size=1)
        self.edge_pub = rospy.Publisher(
            robot_name+"/visualization/edge", MarkerArray, queue_size=1)
        self.twist_pub = rospy.Publisher(
            robot_name+"/mobile_base/commands/velocity", Twist, queue_size=1)
        self.goal_pub = rospy.Publisher(
            robot_name+"/goal", PoseStamped, queue_size=1)
        self.panoramic_view_pub = rospy.Publisher(
            robot_name+"/panoramic", Image, queue_size=1)
        self.topomap_pub = rospy.Publisher(
            robot_name+"/topomap", TopoMapMsg, queue_size=1)
        self.unexplore_direction_pub = rospy.Publisher(
            robot_name+"/ud", UnexploredDirectionsMsg, queue_size=1)
        self.start_pub = rospy.Publisher(
            "/start_exp", String, queue_size=1)

        network = rospy.get_param("~network")
        self.network_gpu = rospy.get_param("~platform")
        if network in PRETRAINED:
            state = load_url(PRETRAINED[network], model_dir= os.path.join(path, "data/networks"))
        else:
            state = torch.load(network)
        net_params = get_net_param(state)

        torch.cuda.empty_cache()
        
        self.net = init_network(net_params)
        self.net.load_state_dict(state['state_dict'])

        self.net.cuda(self.network_gpu)
        self.net.eval()

        normalize = transforms.Normalize(
            mean=self.net.meta['mean'],
            std=self.net.meta['std']
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        self.last_vertex = -1
        self.current_node = None
        self.last_nextmove = 0

        self.laserProjection = LaserProjection()
        self.pcd_queue = Queue(maxsize=10)
        self.detect_loop = 0
        self.grid_map_ready = 0
        self.init_map_angle_ready = 0
        self.tf_transform_ready = 0

        self.topomap_meet = 0

        self.no_place_to_go = 0
        self.goal = np.array([10000.0, 10000.0])

        self.erro_count = 0

        self.tf_listener = tf.TransformListener()
        self.tf_listener2 = tf.TransformListener()
        self.tf_transform = None
        self.rotation = None

        self.actoinclient = actionlib.SimpleActionClient(robot_name+'/move_base', MoveBaseAction)
        self.trajectory_point = None
        self.trajectory_rate = Rate(0.3)
        self.trajectory_length = 0
        self.vertex_on_path = []
        self.reference_vertex = None

        self.start_time = time.time()

        rospy.Subscriber(
            robot_name+"/odom", Odometry, self.map_odom_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/odom", Odometry, self.loop_detect_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/panoramic", Image, self.map_panoramic_callback, queue_size=1)
        rospy.Subscriber(
            robot_name+"/map", OccupancyGrid, self.map_grid_callback, queue_size=1)
        for robot in robot_list:
            rospy.Subscriber(
                robot+"/topomap", TopoMapMsg, self.topomap_callback, queue_size=1)
            rospy.Subscriber(
                robot+"/ud", UnexploredDirectionsMsg, self.unexplored_directions_callback, robot, queue_size=1)
        rospy.Subscriber(
            robot_name+"/move_base/status", GoalStatusArray, self.move_base_status_callback, queue_size=1)
        
        self.actoinclient.wait_for_server()


    def get_move_goal(self, robot_name, last_pose, nextmove, basic_length=4)-> MoveBaseGoal():
        goal_message = MoveBaseGoal()
        goal_message.target_pose.header.frame_id = robot_name + "/map"
        goal_message.target_pose.header.stamp = rospy.Time.now()
        euler = R.from_quat(self.rotation).as_euler('xyz', degrees=True)[2]
        orientation = R.from_euler('z', nextmove+euler, degrees=True).as_quat()
        goal_message.target_pose.pose.orientation.x = orientation[0]
        goal_message.target_pose.pose.orientation.y = orientation[1]
        goal_message.target_pose.pose.orientation.z = orientation[2]
        goal_message.target_pose.pose.orientation.w = orientation[3]

        pose = Point()
        nextmove = math.radians(nextmove)
        theta = math.radians(euler)
        x = last_pose.x + basic_length * np.cos(nextmove)
        y = last_pose.y + basic_length * np.sin(nextmove)
        goal = np.array([x, y])
        pose.x = x*np.cos(theta)-y*np.sin(theta) + self.tf_transform[0]
        pose.y = y*np.cos(theta)+x*np.sin(theta) + self.tf_transform[1]
        goal_message.target_pose.pose.position = pose

        return goal_message, goal

    def get_goal_marker(self, robot_name, last_pose, nextmove, basic_length=4) -> PoseStamped():
        goal_marker = PoseStamped()
        goal_marker.header.frame_id = robot_name + "/map"
        goal_marker.header.stamp = rospy.Time.now()
        euler = R.from_quat(self.rotation).as_euler('xyz', degrees=True)[2]
        orientation = R.from_euler('z', nextmove+euler, degrees=True).as_quat()
        goal_marker.pose.orientation.x = orientation[0]
        goal_marker.pose.orientation.y = orientation[1]
        goal_marker.pose.orientation.z = orientation[2]
        goal_marker.pose.orientation.w = orientation[3]

        pose = Point()
        nextmove = math.radians(nextmove)
        theta = math.radians(euler)
        x = last_pose.x + basic_length * np.cos(nextmove)
        y = last_pose.y + basic_length * np.sin(nextmove)
        pose.x = x*np.cos(theta)-y*np.sin(theta) + self.tf_transform[0]
        pose.y = y*np.cos(theta)+x*np.sin(theta) + self.tf_transform[1]
        goal_marker.pose.position = pose

        return goal_marker
    
    def map_odom_callback(self, data):
        self.pose = data.pose
        if self.init_map_angle_ready == 0:
            self.map_orientation = [data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w]
            self.map_angle = R.from_quat(self.map_orientation).as_euler('xyz', degrees=True)[2]
            self.map.offset_angle = self.map_angle
            self.init_map_angle_ready = 1
        tmptimenow = rospy.Time.now()
        self.tf_listener2.waitForTransform(robot_name+"/map", robot_name+"/odom", tmptimenow, rospy.Duration(0.1))
        try:
            self.tf_transform, self.rotation = self.tf_listener2.lookupTransform(robot_name+"/map", robot_name+"/odom", tmptimenow)
            self.tf_transform_ready = 1
        except:
            pass
        current_pose = data.pose.pose.position
        current_pose = np.array([current_pose.x, current_pose.y])
        if np.sqrt(np.sum(np.square(current_pose-self.goal))) < 0.5:
            print("no place to go")
            self.no_place_to_go = 1

    
    def loop_detect_callback(self, data):
        position = self.pose.pose.position
        position = np.array([position.x, position.y])
        if self.detect_loop:
            for i in range(len(self.map.vertex)):
                meeted = list()
                vertex = self.map.vertex[i]
                for j in range(len(vertex.navigableDirection)-1, -1, -1):
                    unexplored = vertex.frontierPoints[j]
                    if np.sqrt(np.sum(np.square(position-unexplored))) < 3:
                        meeted.append(j)
                for index in meeted:
                    del self.map.vertex[i].navigableDirection[index]
                    del self.map.vertex[i].frontierDistance[index]
                    del self.map.vertex[i].frontierPoints[index]
                    ud_message = UnexploredDirectionsMsg()
                    ud_message.robot_name = vertex.robot_name
                    ud_message.vertexID = vertex.id
                    ud_message.directionID = index
                    self.unexplore_direction_pub.publish(ud_message)
            rr = 150
            for i in range(len(self.map.vertex)):
                vertex = self.map.vertex[i]
                vertex_position = np.array([vertex.pose.pose.position.x, vertex.pose.pose.position.y])
                if vertex.navigableDirection:
                    if np.sqrt(np.sum(np.square(position-vertex_position))) < 100:
                        location = [0, 0]
                        shape = self.global_map.shape
                        location[0] = self.current_loc[0] + int((self.pose.pose.position.x - vertex_position[0])/0.05)
                        location[1] = self.current_loc[1] - int((self.pose.pose.position.y - vertex_position[1])/0.05)
                        temp_map = self.global_map[max(location[0]-rr,0):min(location[0]+rr,shape[0]), max(location[1]-rr,0):min(location[1]+rr, shape[1])]
                        self.map.vertex[i].localMap = temp_map
                        old_ud = self.map.vertex[i].navigableDirection
                        new_ud = self.map.vertex[i].navigableDirection
                        delete_list = []
                        for j in range(len(old_ud)):
                            not_deleted = 1
                            for uds in new_ud:
                                if abs(old_ud[j]-uds) < 5:
                                    not_deleted = 0
                            if not_deleted == 1:
                                delete_list.append(j)
                        for uds in delete_list:
                            ud_message = UnexploredDirectionsMsg()
                            ud_message.robot_name = self.map.vertex[i].robot_name
                            ud_message.vertexID = self.map.vertex[i].id
                            ud_message.directionID = uds
                            self.unexplore_direction_pub.publish(ud_message)
                if np.linalg.norm(position-vertex_position) < 2.5:
                    new_vertex_on_path = 1
                    for svertex in self.vertex_on_path:
                        if svertex.robot_name == vertex.robot_name and svertex.id == vertex.id:
                            new_vertex_on_path = 0
                    if new_vertex_on_path == 1 and vertex.robot_name!= self.reference_vertex.robot_name and vertex.id != self.reference_vertex.id:
                        self.vertex_on_path.append(vertex)
                if np.linalg.norm(self.goal - vertex_position) < 3:
                    self.vertex_on_path.append(vertex)
        if len(self.map.vertex) != 0:
            topomap_message = TopomapToMessage(self.map)
            self.topomap_pub.publish(topomap_message)
        if self.reference_vertex == None:
            self.reference_vertex = self.current_node
        if len(self.vertex_on_path) >= 3:
            self.no_place_to_go = 1
            print(robot_name, "no place to go")
            self.vertex_on_path = []
            self.reference_vertex = self.current_node
    
    def create_panoramic_callback(self, image1, image2, image3, image4):
        img1 = self.cv_bridge.imgmsg_to_cv2(image1, desired_encoding="rgb8")
        img2 = self.cv_bridge.imgmsg_to_cv2(image2, desired_encoding="rgb8")
        img3 = self.cv_bridge.imgmsg_to_cv2(image3, desired_encoding="rgb8")
        img4 = self.cv_bridge.imgmsg_to_cv2(image4, desired_encoding="rgb8")
        panoram = [img1, img2, img3, img4]
        self.panoramic_view = np.hstack(panoram)
        cv2.imwrite("/home/zzl/zzlWorkspace/debug/panormaic.jpg", cv2.cvtColor(self.panoramic_view, cv2.COLOR_BGR2RGB))
        cv2.imwrite("/home/zzl/zzlWorkspace/debug/1.jpg", cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        cv2.imwrite("/home/zzl/zzlWorkspace/debug/3.jpg", cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
        image_message = self.cv_bridge.cv2_to_imgmsg(self.panoramic_view, encoding="rgb8")
        image_message.header.stamp = rospy.Time.now()
        image_message.header.frame_id = robot_name+"/odom"
        self.panoramic_view_pub.publish(image_message)
    
    def map_panoramic_callback(self, panoramic):
        start_msg = String()
        start_msg.data = "Start!"
        self.start_pub.publish(start_msg)
        
        offset = 6
        panoramic_view = self.cv_bridge.imgmsg_to_cv2(panoramic, desired_encoding="rgb8")
        feature = cal_feature(self.net, panoramic_view, self.transform, self.network_gpu)
        current_pose = self.pose
        vertex = Vertex(robot_name, id=-1, pose=current_pose, descriptor=feature)
        self.last_vertex, self.current_node, matched_flag = self.map.add(vertex, self.last_vertex, self.current_node)
        
        if matched_flag==0:
            while self.grid_map_ready==0 or self.tf_transform_ready==0:
                time.sleep(0.5)
            
            # set and publish the topological map visulization markers

            localMap = self.grid_map
            self.map.vertex[-1].localMap = localMap
            self.detect_loop = 0
            picked_vertex_id = self.map.upgradeFrontierPoints(self.last_vertex)

            marker_array = MarkerArray()
            marker_message = set_marker(robot_name, len(self.map.vertex), self.map.vertex[0].pose, action=Marker.DELETEALL)
            marker_array.markers.append(marker_message)
            self.marker_pub.publish(marker_array)
            marker_array = MarkerArray()
            markerid = 0
            for vertex in self.map.vertex:
                if vertex.robot_name != robot_name:
                    marker_message = set_marker(robot_name, markerid, vertex.pose)
                else:
                    marker_message = set_marker(robot_name, markerid, vertex.pose, color=(1,0,0))
                marker_array.markers.append(marker_message)
                markerid += 1
                direction_marker_id = 0
            edge_array = MarkerArray()
            for edge in self.map.edge:
                num_count = 0
                poses = []
                for vertex in self.map.vertex:
                    if (edge.link[0][0]==vertex.robot_name and edge.link[0][1]==vertex.id) or (edge.link[1][0]==vertex.robot_name and edge.link[1][1]==vertex.id):
                        poses.append(vertex.pose.pose.position)
                        num_count += 1
                    if num_count == 2:
                        edge_message = set_edge(robot_name, edge.id, poses)
                        edge_array.markers.append(edge_message)
                        break
            self.marker_pub.publish(marker_array)
            self.edge_pub.publish(edge_array)
            # print(' ')

            position = current_pose.pose.position
            orientation = current_pose.pose.orientation
            orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
            euler = R.from_quat(orientation).as_euler('xyz', degrees=True)[2]
            goal_message, _ = self.get_move_goal(robot_name, position, euler, 0)
            goal_marker = self.get_goal_marker(robot_name, position, euler, 0)
            self.actoinclient.send_goal(goal_message)
            self.goal_pub.publish(goal_marker)
            

            navigableDirection = self.map.vertex[picked_vertex_id].navigableDirection
            nextmove = 0
            directionID = 0
            max_dis = 0
            dis_with_other_centers = [0 for i in range(len(self.meeted_robot))]
            dis_scores = [0 for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints))]
            dis_with_vertices = [0 for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints))]
            epos = 0.2
            if len(navigableDirection) == 0:
                self.no_place_to_go = 1
            else:
                for i in range(len(self.map.vertex[picked_vertex_id].frontierPoints)):
                    position_tmp = self.map.vertex[picked_vertex_id].frontierPoints[i]
                    dis_tmp = np.sqrt(np.sum(np.square(position_tmp-self.map.center)))
                    dis_scores[i] += epos * dis_tmp
                    for j in range(len(self.meeted_robot)):
                        dis_with_other_centers[j] = np.sqrt(np.sum(np.square(position_tmp-self.map.center_dict[self.meeted_robot[j]])))
                        dis_scores[i] += dis_with_other_centers[j] * (1-epos)
                    dis_scores[i] += abs(nextmove - self.last_nextmove)
                for i in range(len(dis_scores)):
                    if dis_scores[i] > max_dis:
                        max_dis = dis_scores[i]
                        directionID = i
                nextmove = navigableDirection[directionID]
                if len(self.map.vertex[picked_vertex_id].navigableDirection)!=0:
                    del self.map.vertex[picked_vertex_id].navigableDirection[directionID]
                    ud_message = UnexploredDirectionsMsg()
                    ud_message.robot_name = self.map.vertex[picked_vertex_id].robot_name
                    ud_message.vertexID = self.map.vertex[picked_vertex_id].id
                    ud_message.directionID = directionID
                    self.unexplore_direction_pub.publish(ud_message)
                if len(self.map.vertex[picked_vertex_id].frontierDistance)!=0:
                    basic_length = self.map.vertex[picked_vertex_id].frontierDistance[directionID]
                    del self.map.vertex[picked_vertex_id].frontierDistance[directionID]
                if len(self.map.vertex[picked_vertex_id].frontierPoints)!=0:
                    del self.map.vertex[picked_vertex_id].frontierPoints[directionID]
                self.detect_loop = 1
                
                nextmove += self.map_angle

                self.last_nextmove = nextmove
                goal_message, self.goal = self.get_move_goal(robot_name, current_pose.pose.position, nextmove, basic_length+offset)
                goal_marker = self.get_goal_marker(robot_name, current_pose.pose.position, nextmove, basic_length+offset)
                self.actoinclient.send_goal(goal_message)
                self.goal_pub.publish(goal_marker)

        if self.no_place_to_go:
            find_a_goal = 0
            position = self.pose.pose.position
            position = np.array([position.x, position.y])
            distance_list = []
            for i in range(len(self.map.vertex)):
                temp_position = self.map.vertex[i].pose.pose.position
                temp_position = np.asarray([temp_position.x, temp_position.y])
                distance_list.append(np.linalg.norm(temp_position - position))
            while(distance_list):
                min_dis = min(distance_list)
                index = distance_list.index(min_dis)
                if len(self.map.vertex[index].navigableDirection) != 0:
                    nextmove = self.map.vertex[index].navigableDirection[0]
                    if len(self.map.vertex[index].navigableDirection)!=0:
                        del self.map.vertex[index].navigableDirection[0]
                        ud_message = UnexploredDirectionsMsg()
                        ud_message.robot_name = self.map.vertex[index].robot_name
                        ud_message.vertexID = self.map.vertex[index].id
                        ud_message.directionID = 0
                        self.unexplore_direction_pub.publish(ud_message)
                    if len(self.map.vertex[index].frontierDistance)!=0:
                        basic_length = self.map.vertex[index].frontierDistance[0]
                        del self.map.vertex[index].frontierDistance[0]
                    if len(self.map.vertex[index].frontierPoints)!=0:
                        point = self.map.vertex[index].pose
                        del self.map.vertex[index].frontierPoints[0]
                    self.detect_loop = 1
                    
                    nextmove += self.map_angle

                    goal_message, self.goal = self.get_move_goal(robot_name, point.pose.position, nextmove, basic_length+offset)
                    goal_marker = self.get_goal_marker(robot_name, point.pose.position, nextmove, basic_length+offset)
                    self.actoinclient.send_goal(goal_message)
                    print(robot_name, "find a new goal")
                    self.goal_pub.publish(goal_marker)
                    find_a_goal = 1
                    break
                del distance_list[index]
                if find_a_goal:
                    break
            self.no_place_to_go = 0

    
    def map_grid_callback(self, data):
        self.global_map_info = data.info
        shape = (data.info.height, data.info.width)
        timenow = rospy.Time.now()
        self.tf_listener.waitForTransform(data.header.frame_id, robot_name+"/base_footprint", timenow, rospy.Duration(0.5))
        try:
            tf_transform, rotation = self.tf_listener.lookupTransform(data.header.frame_id, robot_name+"/base_footprint", timenow)
            self.current_loc = [0,0]
            self.current_loc[0] = int((tf_transform[1] - data.info.origin.position.y)/data.info.resolution)
            self.current_loc[1] = int((tf_transform[0] - data.info.origin.position.x)/data.info.resolution)
            range = 120
            self.global_map = np.asarray(data.data).reshape(shape)
            self.grid_map = self.global_map[max(self.current_loc[0]-range,0):min(self.current_loc[0]+range,shape[0]), max(self.current_loc[1]-range,0):min(self.current_loc[1]+range, shape[1])]
            self.grid_map[np.where(self.grid_map==-1)] = 255
            if robot_name == 'robot1':
                self.global_map[np.where(self.global_map==-1)] = 255
                temp = self.global_map[max(self.current_loc[0]-range,0):min(self.current_loc[0]+range,shape[0]), max(self.current_loc[1]-range,0):min(self.current_loc[1]+range, shape[1])]
                temp[np.where(temp==-1)] = 125
                cv2.imwrite("/home/zzl/zzlWorkspace/debug/map.jpg", temp)
                cv2.imwrite("/home/zzl/zzlWorkspace/debug/globalmap.jpg", self.global_map)

            self.grid_map_ready = 1
        except:
            # print("tf listener fails")
            pass
    
    def topomap_callback(self, topomap_message):
        Topomap = MessageToTopomap(topomap_message)
        matched_vertex = list()
        vertex_pair = dict()
        if Topomap.vertex[0].robot_name in self.meeted_robot:
            for vertex in Topomap.vertex:
                if vertex.robot_name!=robot_name:
                    if vertex.id not in self.vertex_dict[vertex.robot_name]:
                        self.map.vertex.append(vertex)
                        self.vertex_dict[vertex.robot_name].append(vertex.id)
                        edge = Topomap.edge[-1]
                        edge.id = self.map.edge_id
                        self.map.edge_id += 1
                        self.map.edge.append(edge)
                        break
        else:
            for vertex in Topomap.vertex:
                if vertex.robot_name not in self.map.center_dict.keys():
                    self.map.center_dict[vertex.robot_name] = np.array([0.0, 0.0])
                if vertex.robot_name not in self.vertex_dict.keys():
                    self.vertex_dict[vertex.robot_name] = list()
                elif (vertex.id in self.vertex_dict[vertex.robot_name]) or (vertex.robot_name==robot_name):
                    pass
                else:
                    for svertex in self.map.vertex:
                        if svertex.robot_name == vertex.robot_name:
                            pass
                        else:
                            score = np.dot(vertex.descriptor.T, svertex.descriptor)
                            if score > 0.75:
                                print("matched:", svertex.robot_name, svertex.id, vertex.robot_name, vertex.id, score)
                                if vertex.robot_name not in self.meeted_robot:
                                    self.meeted_robot.append(vertex.robot_name)
                                    self.relative_position[vertex.robot_name] = [svertex.pose.pose.position.x-vertex.pose.pose.position.x, svertex.pose.pose.position.y-vertex.pose.pose.position.y, svertex.pose.pose.position.z-vertex.pose.pose.position.z]
                                matched_vertex.append(vertex)
                                self.vertex_dict[vertex.robot_name].append(vertex.id)
                                vertex_pair[vertex.id] = svertex.id
            if len(matched_vertex)!=0:
                for edge in Topomap.edge:
                    if edge.link[0][1] in self.vertex_dict[edge.link[0][0]]:
                        if edge.link[1][1] in self.vertex_dict[edge.link[1][0]] or edge.link[1][0]==robot_name:
                            pass
                        else:
                            edge.link[0][0] = robot_name
                            edge.link[0][1] = vertex_pair[edge.link[0][1]]
                            edge.id = self.map.edge_id
                            self.map.edge_id += 1
                            self.map.edge.append(edge)
                    elif edge.link[1][1] in self.vertex_dict[edge.link[1][0]]:
                        if edge.link[0][1] in self.vertex_dict[edge.link[0][0]] or edge.link[0][0]==robot_name:
                            pass
                        else:
                            edge.link[1][0] = robot_name
                            edge.link[1][1] = vertex_pair[edge.link[1][1]]
                            edge.id = self.map.edge_id
                            self.map.edge_id += 1
                            self.map.edge.append(edge)
                    else:
                        edge.id = self.map.edge_id
                        self.map.edge_id += 1
                        self.map.edge.append(edge)
                for vertex in Topomap.vertex:
                    if vertex.id in self.vertex_dict[vertex.robot_name]:
                        pass
                    else:
                        self.map.vertex.append(vertex)
                        point = self.map.center_dict[vertex.robot_name]
                        number = len(self.vertex_dict[vertex.robot_name])
                        point[0] = (point[0]*number + vertex.pose.pose.position.x) / (number+1)
                        point[1] = (point[1]*number + vertex.pose.pose.position.y) / (number+1)
                        self.map.center_dict[vertex.robot_name] = point
                        self.vertex_dict[vertex.robot_name].append(vertex.id)

    def unexplored_directions_callback(self, data, rn):
        for i in range(len(self.map.vertex)):
            vertex = self.map.vertex[i]
            if vertex.robot_name == data.robot_name:
                if vertex.id == data.vertexID:
                    try:
                        del self.map.vertex[i].navigableDirection[data.directionID]
                    except:
                        pass
    
    def move_base_status_callback(self, data):
        try:
            status = data.status_list[-1].status
        # print(status)
        
            if status >= 3:
                self.erro_count +=1
            if self.erro_count >= 3:
                self.no_place_to_go = 1
                self.erro_count = 0
        except:
            pass

    def trajectory_length_callback(self, data):
        if self.trajectory_point == None:
            self.trajectory_point = data.markers[2].points[-1]
        temp_position = data.markers[2].points[-1]
        point1 = np.asarray([self.trajectory_point.x, self.trajectory_point.y])
        point2 = np.asarray([temp_position.x, temp_position.y])
        self.trajectory_length += np.linalg.norm(point1 - point2)
        self.trajectory_point = temp_position
        print(robot_name, "length", self.trajectory_length)


if __name__ == '__main__':
    time.sleep(5)
    rospy.init_node('topological_map')
    robot_name = rospy.get_param("~robot_name")
    robot_num = rospy.get_param("~robot_num")
    print(robot_name, robot_num)

    robot_list = list()
    for rr in range(robot_num):
        robot_list.append("robot"+str(rr+1))
    robot_list.remove(robot_name)
    node = MapNode(robot_name, robot_list)
    print("node init done")

    robot1_image1_sub = message_filters.Subscriber(robot_name+"/camera1/image_raw", Image)
    robot1_image2_sub = message_filters.Subscriber(robot_name+"/camera2/image_raw", Image)
    robot1_image3_sub = message_filters.Subscriber(robot_name+"/camera3/image_raw", Image)
    robot1_image4_sub = message_filters.Subscriber(robot_name+"/camera4/image_raw", Image)
    ts = message_filters.TimeSynchronizer([robot1_image1_sub, robot1_image2_sub, robot1_image3_sub, robot1_image4_sub], 10)
    ts.registerCallback(node.create_panoramic_callback)

    rospy.spin()