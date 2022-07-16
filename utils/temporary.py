
'''
trav_map_erosion = 2

with open (os.path.join(get_scene_path(scene_id), 'floors.txt'), 'r') as f:
    floors = sorted(list(map(float, f.readlines())))
    print("floor heights", floors)

for f in range(len(floors)):
    trav_map = cv2.imread(os.path.join(get_scene_path(scene_id), 'floor_trav_{}.png'.format(f)))
    obstacle_map = cv2.imread(os.path.join(get_scene_path(scene_id), 'floor_{}.png'.format(f)))
    #trav_map = cv2.resize(trav_map, (trav_map_size, trav_map_size))
    #obstacle_map = cv2.resize(obstacle_map, (trav_map_size, trav_map_size))
    trav_map[obstacle_map == 0] = 0
    trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
    [map_width, map_height, map_channel] = trav_map.shape
    print(trav_map.shape)
    cv2.flip(trav_map, 0, trav_map)
    cv2.namedWindow('trav_map', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('trav_map', trav_map)
'''
    


'''for _ in range(10):
    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    obj.set_position_orientation(np.random.uniform(low=0, high=2, size=3), [0, 0, 0, 1])'''

'''
    while True:
        toward = cv2.waitKey()
        if toward == 113:
            break
        action = actionmapping(toward, 2)
        print(toward, action)
        turtlebot.apply_action(action)
        s.step()
        rgb = s.renderer.render_robot_cameras(modes=('rgb'))
'''