import argparse
import subprocess
import cv2
import av
import numpy as np
import os
from typing import Optional, Any
import csv
from tqdm import tqdm
import json
import concurrent.futures
import matplotlib.pyplot as plt
import math
from fractions import Fraction

SYNC_VIIW = 'brightNess.png' # not None, visualaze result in refine_brightness_synv
def extract_frames_by_pts(video_path: str, pts_of_marked_frames: list[int], shift_pts: int = 0, out_dir: str = "frames",task_id:int=0):
    os.makedirs(out_dir, exist_ok=True)
    with av.open(video_path) as container:
        video_stream = container.streams.video[0]
        target_pts_list = [pts + shift_pts for pts in sorted(pts_of_marked_frames)]
        for idx, target_pts in tqdm(enumerate(target_pts_list),position=task_id,total=len(target_pts_list),desc=f"extracting frame, task {task_id}"):
            found = False
            container.seek(target_pts,any_frame=False, backward=True, stream=video_stream)
            for frame in container.decode(video_stream):
                if frame.pts is not None and frame.pts >= target_pts:
                    img = frame.to_ndarray(format='bgr24')
                    video_name = os.path.basename(video_path)
                    out_path = os.path.join(out_dir, f"{idx:03d}_{video_name[0:3]}.jpg")
                    cv2.imwrite(out_path, img)
                    # print(f"\r saved: {out_path} (PTS={frame.pts}, target={target_pts})",end='')
                    found = True
                    break
            if not found:
                print(f"fialed to find frames with PRS >= target PTS={target_pts},skip")

def plot_jump_times_on_brightness(video_paths: list[str], jump_times: dict, labels: Optional[list] = None):
    """
    :param video_paths: video pathes
    :param jump_times: {video_path: jump time in pts}
    :param labels: labels of curve, optional
    """
    if labels is None:
        labels = [os.path.basename(v) for v in video_paths]
    plt.figure(figsize=(12, 5))
    for idx, video_path in enumerate(video_paths):
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            assert stream.time_base is not None
            times = []
            brightness = []
            for frame in container.decode(stream):
                if frame.key_frame:
                    # t = frame.pts * time_base
                    gray = frame.to_ndarray(format='gray')
                    mean_brightness = gray.mean()
                    times.append(frame.pts)
                    brightness.append(mean_brightness)
            container.close()
            plt.plot(times, brightness, label=labels[idx] if idx < len(labels) else f"Video {idx+1}")
            # 标注 jump_time
            jt = jump_times.get(video_path, None)
            if jt is not None:
                plt.axvline(jt, color=plt.gca().lines[-1].get_color(), linestyle='--', alpha=0.7)
                # plt.text(jt_sec, max(brightness), f"{labels[idx]} jump", rotation=90, va='top', ha='right', fontsize=8)
        except Exception as e:
            print(f"{video_path} failed: {e}")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Brightness (Key Frames)")
    plt.title("Brightness Over Time (Key Frames Only) with Jump Times")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
def find_first_brightness_jump(video_path: str, threshold: float = 30.0, fine_range: float = 3,start:Optional[float]=None,stop:Optional[float]=None)->tuple[int,Fraction,int,int,Fraction, int]|tuple[None,None,None,None,None,None]:
    """
    video with CFR
    args:
        - fine_range: time range for fine search, in seconds
        - start: searching start, in seconds
        - stop: searching stop, in seconds
    returns:
        - (int,int):jump frame idx, jump time in ms .When failed, (-1,-1).
    """
    with av.open(video_path) as container:
        jump_frame_pts = -1
        pts_interval = None
        num_frames = None
        time_base = None
        first_pts = None
        video_stream = container.streams.video[0]
        fps = video_stream.average_rate
        time_base = video_stream.time_base
        num_frames = video_stream.frames
        for frame in container.decode(video_stream):
            if first_pts is None:
                first_pts = frame.pts
                continue
            pts_interval = frame.pts - first_pts
            break
        assert fps is not None and time_base is not None and pts_interval is not None and num_frames is not None and first_pts is not None,"faied with pts and frame info"
        fine_range = int(round(fine_range*fps*pts_interval)+1)
        container.seek(0)
        prev_brightness = None
        jump_frame_idx = -1
        video_stream.codec_context.skip_frame = "NONKEY"
        fine_search_start = None
        fine_search_end = None
        if start is not None:
            start_pts_toseek = int(start / time_base)
            container.seek(start_pts_toseek, any_frame=False, backward=True, stream=video_stream)
        if stop is not None:
            stop_pts = int(stop / time_base)
            for frame in container.decode(video_stream):
                img = frame.to_ndarray(format='gray')
                pts = frame.pts
                if  pts > stop_pts:
                    break
                mean_brightness = img.mean()
                if prev_brightness is not None:
                    if mean_brightness - prev_brightness > threshold:
                        fine_search_start = max(0, pts - fine_range)
                        fine_search_end = pts + fine_range
                        break
                prev_brightness = mean_brightness
        else:
            for frame in container.decode(video_stream):
                img = frame.to_ndarray(format='gray')
                pts = frame.pts
                mean_brightness = img.mean()
                if prev_brightness is not None:
                    if mean_brightness - prev_brightness > threshold:
                        fine_search_start = max(0, pts - fine_range)
                        fine_search_end = min(pts + fine_range,num_frames)
                        break
                prev_brightness = mean_brightness
        if not (fine_search_start is not None and fine_search_end is not None):
            print(f"fail at coarse find")
            return None,None,None,None,None,None
        
        video_stream.codec_context.skip_frame = "DEFAULT"  # 确保逐帧解码
        container.seek(fine_search_start, any_frame=False, backward=True, stream=video_stream)
        brightness_list = []
        pts_indices = []
        for frame in container.decode(video_stream):
            # img = frame.to_ndarray(format='gray')
            pts_indices.append(frame.pts)
            gray = frame.to_ndarray(format='gray')
            brightness_list.append(gray.mean())
    
        brightness_arr = np.array(brightness_list)
        delta_brightness = np.diff(brightness_arr)
        if len(delta_brightness) > 0:
            max_jump_idx = np.argmax(delta_brightness)
            jump_frame_pts = pts_indices[max_jump_idx]
        else:
            jump_frame_pts = -1

        if jump_frame_pts != -1:
            jump_frame_idx = int(round((jump_frame_pts - first_pts)/pts_interval))
            jump_frame_in_s = jump_frame_pts * time_base
            return jump_frame_pts,jump_frame_in_s, jump_frame_idx,pts_interval,time_base, num_frames
        else:
            return None,None,None,None,None,None
    print(f'fail to read video {video_path}')
    return None,None,None,None,None,None

def refine_brightness_synv(video_paths: list[str], ref_video: str,original_anchor_pts:dict[str,int],start_pts:dict[str,int],stop_pts:dict[str,int], view:Optional[str]=SYNC_VIIW)->dict[str,int]:
    """
    :param video_paths: 
    :param ref_video: 
    :param shift_pts: 
    :param jump_pts: 
    :param refine_range: 
    :return: dict {video_path: (pts_list, brightness_list)}
    """

    def read_brightness_curve(v:str,start_pts:int,end_pts:int):
        try:
            pts_list = []
            brightness_list = []
            with av.open(v,mode='r') as container:
                video_stream = container.streams.video[0]
                container.seek(start_pts, any_frame=False, backward=True, stream=video_stream)
                for frame in container.decode(video_stream):
                    pts = frame.pts
                    if pts is None:
                        continue
                    elif pts > end_pts:
                        break
                    elif pts < start_pts:
                        continue
                    gray = frame.to_ndarray(format='gray')
                    brightness = gray.mean()
                    pts_list.append(pts)
                    brightness_list.append(brightness)
            return v, pts_list, brightness_list
        except Exception as e:
            print(f"{v} failed: {e}")
            return v, None, None

    brightness_curves = {}
    pts_lists = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(read_brightness_curve, v,start_pts[v],stop_pts[v]): v for v in video_paths}
        for future in concurrent.futures.as_completed(futures):
            v, pts_list, brightness_list = future.result()
            if pts_list is not None and brightness_list is not None:
                pts_lists[v] = pts_list
                brightness_curves[v] = brightness_list

    # 以ref_video为基准
    ref_pts_list = pts_lists[ref_video]
    ref_brightness = np.asarray(brightness_curves[ref_video])
    refined_shfits = {}
    for v in brightness_curves:
        if v == ref_video:
            refined_shfits[ref_video] = 0
            continue
        pts_list = pts_lists[v]
        brightness = brightness_curves[v]
        min_len = min(len(ref_brightness), len(brightness))
        if min_len < 3:
            print(f"{v} 曲线太短，跳过互相关校准")
            continue
        ref_b = ref_brightness[:min_len]
        b = np.array(brightness[:min_len])
        # 归一化
        ref_b = (ref_b - ref_b.mean()) / (ref_b.std() + 1e-8)
        b = (b - b.mean()) / (b.std() + 1e-8)
        corr = np.correlate(b, ref_b, mode='full')
        shift = np.argmax(corr) - (min_len - 1)
        # pts_interval = np.median(np.diff(pts_list))
        pts_interval = pts_list[1] - pts_list[0]
        shift_pts = int(round(shift * pts_interval))

        original_offset = original_anchor_pts[v] - original_anchor_pts[ref_video]
        new_offset = original_offset + shift_pts
        print(f"{os.path.basename(v)}: original_offset={original_offset}, shift={shift}, new_offset={new_offset}")
        refined_shfits[v] = new_offset
    if view:
        plt.figure(figsize=(12, 5))
        color_map = {}
        colors = plt.colormaps.get_cmap('tab20')
        for idx, v in enumerate(brightness_curves):
            color = colors(idx)
            color_map[v] = color
            pts_list = np.array(pts_lists[v])
            brightness = np.array(brightness_curves[v])
            shift = refined_shfits.get(v, 0)
            shifted_pts = pts_list - shift
            plt.plot(shifted_pts, brightness, label=os.path.basename(v), color=color)
            # 标注 original_anchor_pts 到可视化图上
            anchor_pts = original_anchor_pts[v] - shift
            if v == ref_video:
                plt.axvline(anchor_pts, color=color, linestyle='-', linewidth=2, label=f"{os.path. basename(v)} (ref jump)")
                plt.text(anchor_pts, np.max(brightness), "REF", color=color, fontsize=10,  fontweight='bold', va='bottom', ha='right')
            else:
                plt.axvline(anchor_pts, color=color, linestyle='--', alpha=0.7, label=f"{os.path.  basename(v)} jump")
                # plt.text(nchor_pts, np.max(brightness), "jump", color=color, fontsize=8, va='bottom',     ha='right')

        plt.xlabel("Aligned PTS")
        plt.ylabel("Mean Brightness")
        plt.title("Brightness Curves After Refined Shift Alignment")
        # plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(view)
    return refined_shfits

def sync_videos_by_brightness_jump(video_paths: list, ref_video: str, threshold: float = 30.0, fine_range: float = 3.0,start:Optional[float]=None, stop:Optional[float]=None):
    """
    param video_paths: video pathes
    param ref_video: anchor video path
    param threshold: on light changes
    fine_range: fine search and refine range, inseconds
    :return: {video_path: shift_index},{video_path: jump_nidex},{video_path: shift_time},{video_path: jump_time}
    """
    jump_ptses = {}
    jump_times = {}
    jump_indexes = {}
    pts_internals = {}
    time_bases = {}
    refine_start_pts = {}
    refine_stop_pts = {}
    print("finding")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_video = {executor.submit(find_first_brightness_jump, v, threshold,fine_range,start,stop): v for v in video_paths}
        for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(video_paths)):
            v = future_to_video[future]
            pts,time,index,pts_internal,time_base, num_frames = future.result()
            if pts is None or fine_range is None or time_base is None or num_frames is None or pts_internal is None:
                print(f"{v} fail to finds light jump")
                continue
            jump_ptses[v] = pts
            jump_indexes[v] = index
            jump_times[v] = time
            pts_internals[v] = pts_internal
            time_bases[v] = time_base
            refine_start_pts[v] = max(0, int(math.floor(pts - fine_range / time_base)))
            refine_stop_pts[v] = min(num_frames*pts_internal, int(math.ceil(pts + fine_range / time_base)))
            
    ref_pts = jump_ptses.get(ref_video, None)
    # ref_index = jump_indexes.get(ref_video, None)
    # ref_time = jump_times.get(ref_video, None)
    if ref_pts is None or ref_pts == -1:
        print(f"refer video {ref_video} no light jump, file")
        return None, None, None, None, None, None
    shift_pts = {v: t-ref_pts if t != -1 else None for v, t in jump_ptses.items()}
    refined_shfit_pts = refine_brightness_synv(video_paths,ref_video,jump_ptses,refine_start_pts,refine_stop_pts)

    refined_shift_times = {}
    refined_shift_indexes = {}
    for v in video_paths:
        if v == ref_video:
            refined_shift_times[v] = 0
            refined_shift_indexes[v] = 0
        else:
            shift_pts_val = refined_shfit_pts.get(v,None)
            pts_interval = pts_internals.get(v,None)
            time_base = time_bases.get(v, None)
            try:
                if pts_interval != 0 and pts_interval != -1:
                    shift_index = int(round(shift_pts_val / pts_interval)) # type: ignore
                else:
                    shift_index = 0
                shift_time = shift_pts_val * time_base # type: ignore
                refined_shift_times[v] = shift_time
                refined_shift_indexes[v] = shift_index
            except Exception as e:
                print(f"Error calculating refined shift for {v}: {e}")
    
    return shift_pts,jump_ptses,refined_shift_times, jump_times,refined_shift_indexes,jump_indexes

def main():
    parser = argparse.ArgumentParser(description="Video brightness analysis and frame extraction tool")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-v","--video", help="Path to the video file")
    parent_parser.add_argument('-o', "--out_dir", default="frames", help="Output image folder")
    parent_parser.add_argument("-r", "--refer_video", type=str, default=None, help="Reference camera for time alignment")
    parent_parser.add_argument("--start", type=float, default=None, help="Approximate alignment start time (in seconds)")
    parent_parser.add_argument("--stop", type=float, default=None, help="Approximate alignment stop time (in seconds)")
    parent_parser.add_argument("--threshold", type=float, default=None, help="threshold of lights change in rough light sync")
    parent_parser.add_argument("--fine_range", type=float, default=None, help="half the range of gine light sync (in seconds)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Plot brightness curves for all videos in a folder
    parser_brightness = subparsers.add_parser("brightness", help="Plot brightness curves for all videos in a folder",parents=[parent_parser])

    # 2. Extract frames from video based on a CSV time list
    parser_extract = subparsers.add_parser("extract", help="Extract frames from video based on a CSV time list",parents=[parent_parser])
    parser_extract.add_argument("-c","--csv", help="Path to the CSV file containing frame times")

    args = parser.parse_args()

    # 构造参数字典，只包含有值的参数
    arg_keys = ["start", "stop", "threshold", "fine_range"]
    arg_dict = {k: getattr(args, k) for k in arg_keys if getattr(args, k) is not None}

    shift_json_path = os.path.join(args.out_dir, "shift_times.json")
    if args.command == "brightness":
        video_files = [os.path.join(args.video, f) for f in os.listdir(args.video)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if len(video_files)==0:
            print("fial to find videos")
            return
        if args.refer_video is not None:
            ref_v = os.path.join(os.path.dirname(video_files[0]),os.path.basename(args.refer_video))
            assert ref_v in video_files,f"refering video {ref_v} not in args.video {args.video}"
        else:
            ref_v = video_files[0]
            
        if os.path.exists(shift_json_path):
            with open(shift_json_path, "r", encoding="utf-8") as f:
                to_load = json.load(f)
                jump_ptses = to_load['jump_ptses']
            print(f"loaded jump_times from {shift_json_path}")
        else:
            refined_shift_ptses,jump_ptses,refined_shift_times, jump_times,refined_shift_indexes,jump_indexes = sync_videos_by_brightness_jump(video_files, ref_v, **arg_dict)
            to_save = {
                'refined_shift_ptses': refined_shift_ptses,
                'jump_ptses': jump_ptses,
                'refined_shift_times': {str(k):str(v) for k,v in refined_shift_times.items()} if refined_shift_times is not None else None,
                'jump_times': {str(k):str(v) for k,v in jump_times.items()} if jump_times is not None else None,
                'refined_shift_indexes': refined_shift_indexes,
                'jump_indexes': jump_indexes
            }
            with open(shift_json_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
            print(f"saving jump_times to {shift_json_path}")
        assert jump_ptses is not None,f"fail to find valid jump_ptses"
        plot_jump_times_on_brightness(video_files,jump_times=jump_ptses)

    elif args.command == "extract":
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        time_of_marked_frames_pts = []
        with open(args.csv, newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time_of_marked_frames_pts.append(int(row["fps"]))
        video_files = [os.path.join(args.video, f) for f in os.listdir(args.video)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if args.refer_video is not None:
            ref_v = os.path.join(os.path.dirname(video_files[0]),os.path.basename(args.refer_video))
            assert ref_v in video_files,f"refering video {ref_v} not in args.video {args.video}"
        else:
            ref_v = video_files[0]

        if os.path.exists(shift_json_path):
            with open(shift_json_path, "r", encoding="utf-8") as f:
                to_load = json.load(f)
                refined_shift_ptses = to_load['refined_shift_ptses']
                jump_ptses = to_load['jump_ptses']
                refined_shift_times = to_load['refined_shift_times']
                jump_times = to_load['jump_times']
                refined_shift_indexes = to_load['refined_shift_indexes']
                jump_indexes = to_load['jump_indexes']

            print(f"loaded shift_times from {shift_json_path}")
        else:
            refined_shift_ptses,jump_ptses,refined_shift_times, jump_times,refined_shift_indexes,jump_indexes = sync_videos_by_brightness_jump(video_files, ref_v, **arg_dict)
            to_save = {
                'refined_shift_ptses': refined_shift_ptses,
                'jump_ptses': jump_ptses,
                'refined_shift_times': {str(k):str(v) for k,v in refined_shift_times.items()} if refined_shift_times is not None else None,
                'jump_times': {str(k):str(v) for k,v in jump_times.items()} if jump_times is not None else None,
                'refined_shift_indexes': refined_shift_indexes,
                'jump_indexes': jump_indexes
            }
            with open(shift_json_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f, ensure_ascii=False, indent=2)
            print(f"saving to {shift_json_path}")
        assert refined_shift_ptses is not None and jump_ptses is not None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []  
            for i,v in enumerate(video_files):
                t = refined_shift_ptses[v]
                assert t is not None, f"sync fail with shift time {refined_shift_ptses}"
                futures.append(executor.submit(extract_frames_by_pts, v, time_of_marked_frames_pts, t, args.out_dir,i))
            for f in concurrent.futures.as_completed(futures):
                f.result()

if __name__ == "__main__":
    main()