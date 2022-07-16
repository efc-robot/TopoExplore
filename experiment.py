#!/usr/bin/python3.8
from urllib import response
import rospy
from rospy.timer import Rate, sleep
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray

import numpy as np
import time

map_area = {"small":114800, "large":243100}


class ExpNode:
    def __init__(self, robot_list) -> None:
        self.map = None
        self.start_time = None
        self.end_time = None
        self.exp_time = None
        rospy.Subscriber("/start_exp", String, self.exp_start_callback)
        self.exp_start = 0
        self.exp_done = 0
        print("before callback")
        while(self.exp_start == 0):
            pass
        print("EXP START!")
        self.start_time = time.time()
        rospy.Subscriber(
            "/robot1/cartographerMap", OccupancyGrid, self.map_callback, queue_size=1)
        self.trajectory_point = [None for robot in robot_list]
        self.trajectory_length = np.zeros(len(robot_list))
        self.half_length = None
        for robot in robot_list:
            rospy.Subscriber(
                robot+"/trajectory_node_list", MarkerArray, self.trajectory_length_callback, callback_args=robot, queue_size=1)

    def map_callback(self, data):
        shape = (data.info.height, data.info.width)
        self.map = np.asarray(data.data).reshape(shape)
        map_size = np.count_nonzero(self.map>=0)
        map_size -= np.count_nonzero(self.map==1)
        explore_rate = map_size / 114800
        # print(explore_rate)
        # print(map_size)
        if explore_rate>0.99 and self.exp_done==0:
            self.exp_done = 1
            self.end_time = time.time()
            self.exp_time = self.end_time-self.start_time
            self.half_length = self.trajectory_length
        # if explore_rate>0.99:
        #     print("almost done,", self.end_time-self.start_time, self.trajectory_length)
    
    def trajectory_length_callback(self, data, robot):
        num = int(robot[5:]) - 1
        if self.trajectory_point[num] == None:
            self.trajectory_point[num] = data.markers[2].points[-1]
        temp_position = data.markers[2].points[-1]
        point1 = np.asarray([self.trajectory_point[num].x, self.trajectory_point[num].y])
        point2 = np.asarray([temp_position.x, temp_position.y])
        self.trajectory_length[num] += np.linalg.norm(point1 - point2)
        self.trajectory_point[num] = temp_position
        # print(robot, "length", data.markers[2].header.stamp)

    def exp_start_callback(self, data):
        self.exp_start = 1
        if self.exp_done == 1:
            print("exp done,", self.exp_time, self.half_length)
        

if __name__ == '__main__':
    rospy.init_node('experiment_node')
    robot_num = rospy.get_param("~robot_num")
    print("Experiment start:", robot_num)
    robot_list = list()
    for rr in range(robot_num):
        robot_list.append("robot"+str(rr+1))
    node = ExpNode(robot_list)
    rospy.spin()