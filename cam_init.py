import yaml
import sys
import requests
import argparse

def read_camera_list(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # 假设相机列表存储在 'cameras' 键下
    camera_list = data.get('cameras', [])
    return camera_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera initialization script")
    parser.add_argument('yaml_file', nargs='?', default='cam_ID.yaml', help='Path to the camera ID YAML file')
    parser.add_argument('-f','--flag', default='1', help='Flag value to send to cameras')
    args = parser.parse_args()
    yaml_file = args.yaml_file
    flag = args.flag
    camerasID = read_camera_list(yaml_file)
    cameras = {id: {'ip': f'172.2{id[0]}.1{id[1:]}.51', 'port': '8080'} for id in camerasID}
    # print(cameras)
    querystring = {"p":flag}
    failed = []
    for id, cam in cameras.items():
        print(f"initing {id}")
        try:
            url = f"http://{cam['ip']}:{cam['port']}/gopro/camera/control/wired_usb"
            response = requests.request("GET", url, params=querystring)
            print(response.url)
            print(response.text)
        except Exception as e:
            failed.append(id)
    print(f"fail to init:")
    print(failed)
