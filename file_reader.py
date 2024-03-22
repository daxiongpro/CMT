import pickle
import json
import yaml

file_ = 'data/megvii_data/new_custom_data/megvii_infos_train.pkl'
# file_ = "xx/xxx.json"
# file_ = 'xx/xxx.yaml'

with open(file_, 'rb') as f:
    if file_.endswith('.pkl'):
        data = pickle.load(f)
    elif file_.endswith('.json'):
        data = json.load(f)
    elif file_.endswith('.yaml'):
        data = yaml.safe_load(f)

# print(data)

old_root = '/home/daxiongpro/code/bevfusion'
new_root = '/cpfs/user/zhengyi/code/CMT'
for i, frame in enumerate(data['infos']):
    old_lidar_path = frame['lidar_path']
    new_lidar_path = old_lidar_path.replace(old_root, new_root)
    frame['pts_filename'] = new_lidar_path
    for cam_name, value in frame['cams'].items():
        old_camera_path = value['camera_path']
        new_camera_path = old_camera_path.replace(old_root, new_root)
        value['img_filename'] = new_camera_path
    print("frame {} finished!".format(i))



with open(file_, 'wb') as f:
    if file_.endswith('.pkl'):
        pickle.dump(data, f)
        print('all success !')
    elif file_.endswith('.json'):
        pass
    elif file_.endswith('.yaml'):
        pass