# TopoExplore

The released Version of MR-TopoMap, the supporting open-sourced software package of the paper "MR-TopoMap: Multi-Robot Exploration Based on Topological Map in Communication Restricted
Environment" submitted to RA-L and IROS 2022.

Our video is released at Bilibili.

Features:
- A multi-robot exploration framework based on topological map
- Only newly-built vertices and edges are transferred among robots, reducing the communication traffic
- An exploration strategy based on topological map, achieving higher exploration efficiency than Multi-robot Multi-target Potential Field (MMPF)[1] and Rapidly Random Tree (RRT)[2]

# Platform
- Multiple cars with NVIDIA Jetson Xavier NX board, cameras and Lidar sensor.
- Ubuntu 20.04 (zsh)
- ROS noetic

# Dependency

## Cartographer
Cartographer is a 2D/3D map-building method.
It provides the submaps' and the trajectories' information when building the map. 

We slightly modified the original Cartographer to make it applicable to multi-robot SLAM and exploration.

Please refer to [Cartographer-for-SMMR](https://github.com/efc-robot/Cartographer-for-SMMR) to install the modified Cartographer to ```carto_catkin_ws```

and 

```
source /PATH/TO/CARTO_CATKIN_WS/devel_isolated/setup.bash
```

## Turtlebot3 Description and Simulation
(robot model for simulation)

```
sudo apt install ros-noetic-turtlebot3*
sudo apt install ros-noetic-bfl
pip install future
sudo apt install ros-noetic-teb-local-planner
```


# Simulation in Gazebo 

## Installation
download the SMMR repo(https://github.com/efc-robot/SMMR-Explore), then switch to topoexplore branch and update the submodule
```
git checkout topoexplore
git submodule update --init
```

Build the project
```
catkin_make
source <repo_path>/devel/setup.bash
```

- Template  
```
{env_size}      = 'small' or 'large'
{number_robots} = 'single' or 'two' or 'three'
{method}        = 'rrt' or 'mmpf'
{suffix}        = 'robot' or 'robots' (be 'robot' when number_robots != 'single')
```


### BaseLine: Multi-robot Multi-target Potential Field (MMPF) and Rapidly Random Tree (RRT)
```
roslaunch turtlebot3sim {env_size}_env_{number_robots}_{suffix}.launch
roslaunch turtlebot3sim {number_robots}_{suffix}.launch
roslaunch {method} {number_robots}_{method}_node.launch
```
### Test Baseline

For all the baseline cases, choose "Publish Point" button in the rviz and then click anywhere in the map to start the exploration.

### Run TopoExplore
```
roslaunch turtlebot3sim {env_size}_env_{number_robots}_{suffix}.launch
roslaunch turtlebot3sim {number_robots}_{suffix}_origin.launch
roslaunch ros_topoexplore {number_robots}_{suffix}_topo.launch
```

## References
1. J. Yu, J. Tong, Y. Xu, Z. Xu, H. Dong, T. Yang, and Y. Wang, “Smmrexplore: Submap-based multi-robot exploration system with multi-robot multi-target potential field exploration method,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 8779–8785.
2. H. Umari and S. Mukhopadhyay, “Autonomous robotic exploration based on multiple rapidly-exploring randomized trees,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017, pp. 1396–1402.