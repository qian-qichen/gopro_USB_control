import requests
import concurrent.futures
import threading
import time
import argparse
import yaml
TIME_OUT = 5
def read_camera_list(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # 假设相机列表存储在 'cameras' 键下
    camera_list = data.get('cameras', [])
    return camera_list


def set_camera(cam_url,barrier):
    try:
        barrier.wait()
        
        start_time = time.perf_counter_ns()

        response = requests.request("GET", cam_url)
        response.raise_for_status()

        latency = (time.perf_counter_ns() - start_time) / 1e6
        
        return {
            "conmmand": cam_url,
            "status": "success",
            "latency_ms": latency,
            "response": response.text
        }
    except Exception as e:
        print(e)
        return {
            "conmmand": cam_url,
            "status": "error",
            "error": str(e)
        }
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera initialization script")
    parser.add_argument('yaml_file', nargs='?', default='cam_ID.yaml', help='Path to the camera ID YAML file')
    parser.add_argument('-f','--flag', default='0', help='Flag value to send to cameras')
    args = parser.parse_args()
    yaml_file = args.yaml_file
    flag = args.flag
    camerasID = read_camera_list(yaml_file)
    cameras =  {id: {'start_url': f'http://172.2{id[0]}.1{id[1:]}.51:8080/gopro/camera/shutter/start',
                     'stop_url':f'http://172.2{id[0]}.1{id[1:]}.51:8080/gopro/camera/shutter/stop',} for id in camerasID}
    barrier = threading.Barrier(len(cameras) + 1) 

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras)) as executor:
        # 提交所有任务
        if flag == '1':
            futures = [executor.submit(set_camera, cam['start_url'], barrier) for cam in cameras.values()]
        else:
            futures = [executor.submit(set_camera, cam['stop_url'], barrier) for cam in cameras.values()]

        # 等待所有线程准备就绪（主线程到达屏障）
        barrier.wait()
        trigger_time = time.perf_counter_ns()

        # 获取结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # 打印结果
    print(f"触发时间: {trigger_time} ns")
    for res in results:
        print(f"command {res['conmmand']}: {res['status']}, "
              f"Latency: {res.get('latency_ms', 0):.2f} ms")