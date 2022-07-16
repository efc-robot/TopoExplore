from platform import version
from scipy.spatial.transform import Rotation as R
import numpy as np


def threeD_to_pcd(threeD) -> np.ndarray:
    threeD = np.asarray(threeD[0])
    pcd = threeD[:, :, :3]
    shape = pcd.shape
    pcd.reshape(shape[0]*shape[1], shape[2])
    
    return pcd


def steering_angle(width, bias_pixel, horizontal_fov) -> float:

    '''distance from camera to image plane'''
    L = (width / 2) / np.tan(horizontal_fov / 2 / 180.0 * np.pi)

    return np.arctan((bias_pixel - width / 2) / L)


def get_action_from_angle(angle, time, radius) -> list:

    velocity = radius * angle / time

    return [velocity / 2, -velocity / 2]

def get_panoramic_view(simulator, robot, rotation_angle) -> list:
    current_orientation = robot.get_orientation()
    current_euler = R.from_quat(current_orientation).as_euler('xyz', degrees=True)[2]

    rot = int(360/rotation_angle)

    RGB = []
    Depth = []

    for i in range(rot):
        q = R.from_euler('z', current_euler+i*rotation_angle, degrees=True).as_quat()
        robot.set_orientation(q)
        simulator.step()
        rgb = simulator.renderer.render_robot_cameras(modes=('rgb'))
        RGB.append(rgb[0][:,:,:3])
        threeD = simulator.renderer.render_robot_cameras(modes=('3d'))
        depth = np.asarray(threeD[0][ :, :, 2])
        depth = np.dot(depth, -128).astype(np.uint8)
        Depth.append(depth)
    
    robot.set_orientation(current_orientation)
    simulator.step()
    
    return RGB, Depth