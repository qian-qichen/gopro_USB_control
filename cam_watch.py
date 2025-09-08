import requests
import time
import yaml
import argparse
import json
import os
from rich.live import Live
from rich.table import Table
import logging

TIME_OUT = 5
# SETTING_MEANINGS={}
# def translate_setting(json_str):
#     status = json.loads(json_str)
#     result = {}
#     for key, value in status.items():
#         if key in SETTING_MEANINGS:
#             meaning = SETTING_MEANINGS[key].get(value, f"未知({value})")
#             result[key] = meaning
#         # else: # ignore key with no meaning offered
#         #     result[key] = value
#     return result
# STATUS_MEANINGS={
#     '6':{'meaning':'overheating'}
# }
# def translate_status(json_str):
#     status = json.loads(json_str)
#     result = {}
#     for key, value in status.items():
#         if key in STATUS_MEANINGS:
#             meaning = STATUS_MEANINGS[key].get('meaning')
#             state = STATUS_MEANINGS[key].get(value)
#             result[meaning] = state
#         # else: # ignore key with no meaning offered
#         #     result[key] = value
#     return result


# Setup logging to file

def read_camera_list(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    camera_list = data.get('cameras', [])
    return camera_list
logging.basicConfig(
    filename="camera_events.log",
    filemode="a",
    format="%(asctime)s |%(levelname)s| Camera %(cam_id)s| %(event)s - %(detail)s| %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
def ipPort2id(ipPort:str):
    parts = ipPort.split('.')
    return parts[1][1:]+parts[2][1:]

def log_event(cam_id, event, detail, is_fault=False):
    extra = {"cam_id": cam_id, "event": event, "detail": detail}
    if is_fault:
        logging.error("", extra=extra)
    else:
        logging.info("", extra=extra)

# State tracking for event detection
def query_state(ipPort):
    try:
        url = f"http://{ipPort}/gopro/camera/state"
        response = requests.request("GET", url)
        response.raise_for_status()
        text = response.text
    except Exception as e:
        log_event(ipPort2id(ipPort), "Query", f"Failed to query camera state: {e}", is_fault=True)
        return None
    try:
        json_obj = json.loads(text)
    except json.JSONDecodeError as e:
        log_event(ipPort2id(ipPort), "Query", f"Failed to decode JSON response: {e}", is_fault=True)
        return None
    return json_obj

def detect_and_log_events(cam_id, prev_state, curr_state):
    if not prev_state:
        return
    status_prev = prev_state.get('status', {})
    status_curr = curr_state.get('status', {})
    # Overheating
    if status_prev.get('6', 0) != status_curr.get('6', 0):
        if status_curr.get('6', 0) == 1:
            log_event(cam_id, "Overheating", "Camera is overheating", is_fault=True)
        elif status_prev.get('6', 0) == 1:
            log_event(cam_id, "Overheating", "Camera recovered from overheating")
    # Storage card status
    if status_prev.get('33', -1) != status_curr.get('33', -1):
        storage = storage_status.get(status_curr.get('33', -1), f"unknown ({status_curr.get('33', -1)})")
        is_fault = status_curr.get('33', -1) in [1,2,3,4,8,-1]
        log_event(cam_id, "Storage Card", f"Status changed to {storage}", is_fault=is_fault)
    # Start/Stop encoding (shooting)
    if status_prev.get('10', 0) != status_curr.get('10', 0):
        if status_curr.get('10', 0) == 1:
            log_event(cam_id, "Shooting", "Camera started recording.")
        elif status_prev.get('10', 0) == 1:
            log_event(cam_id, "Shooting", "Camera stopped recording.")

camera_prev_states = {}
def query_state_with_log(ipPort, cam_id):
    state = query_state(ipPort)
    prev_state = camera_prev_states.get(cam_id)

    if state and prev_state is None:
        log_event(cam_id, "Online", "Camera is online")
    if state is None and prev_state is not None:
        log_event(cam_id, "Offline", "Camera is offline", is_fault=True)

    if state is not None and prev_state is not None:
        detect_and_log_events(cam_id, prev_state, state)
    camera_prev_states[cam_id] = state
    return state


'''
6:Overheating
8:Busy
10:Encoding
13:Video Encoding Duration
33:Primary Storage
    -1 	Unknown
    0 	OK
    1 	SD Card Full
    2 	SD Card Removed
    3 	SD Card Format Error
    4 	SD Card Busy
    8 	SD Card Swapped
35:Remaining Video Time
116:USB Controlled
'''
storage_status = {
    -1: 'Unknown',
    0: 'OK',
    1: 'Full',
    2: 'Removed',
    3: 'Format Error',
    4: 'Busy',
    8: 'SD Card Swapped'
}
TFid_to_meaning = {
    0: 'No',
    1: 'Yes',
    -1: 'N/A',
}

def visualize_camera_states(cameras, interval=10):
    def build_table():
        table = Table()
        table.title = "camera state"
        table.add_column("Camera ID", justify="left")
        # table.add_column("IP:Port", justify="left")
        table.add_column("Overheating", justify="center")
        table.add_column("Busy", justify="center")
        table.add_column("Encoding", justify="center")
        table.add_column("Video Encoding Time", justify="center")
        table.add_column("Storage", justify="center")
        table.add_column("Remain Time", justify="center")
        table.add_column("USB Ctrl", justify="center")
        for cam_id, cam_info in cameras.items():
            ipPort = cam_info['ipPort']
            # state = query_state(ipPort)
            state = query_state_with_log(ipPort,cam_id)
            if state is None:
                row = [str(cam_id), ipPort] + ['N/A'] * 6
            else:
                status = state['status']
                overheating = TFid_to_meaning[status.get('6',-1)]
                busy = TFid_to_meaning[status.get('8', -1)]
                encoding = TFid_to_meaning[status.get('10', -1)]
                storage_id = status.get('33', -1)
                storage = storage_status.get(storage_id, f"unknown ({storage_id})")
                video_encoding_time = f"{status.get('13', 'N/A')} s"
                remain_time = f"{status.get('35', 'N/A')} s"
                usb_ctrl = TFid_to_meaning[status.get('116', 0)]
                # row = [str(cam_id), ipPort, overheating, busy,video_encoding_time, encoding, storage, remain_time, usb_ctrl]
                row = [str(cam_id), overheating, busy, encoding,video_encoding_time, storage, remain_time, usb_ctrl]
            table.add_row(*row)
        return table

    with Live(build_table(), refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(interval)
            table = build_table()
            live.update(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera initialization script")
    parser.add_argument('yaml_file', nargs='?', default='cam_ID.yaml', help='Path to the camera ID YAML file')
    parser.add_argument('-i','--interval', type=float, default=1.0, help='Query interval in seconds')
    args = parser.parse_args()
    camerasID = read_camera_list(args.yaml_file)
    cameras = {id: {'ipPort': f'172.2{id[0]}.1{id[1:]}.51:8080'} for id in camerasID}
    visualize_camera_states(cameras, args.interval)