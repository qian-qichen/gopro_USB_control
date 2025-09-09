import cv2
import time
import csv
import json
import argparse
import av
import os
import subprocess

def get_all_pts_ffprobe(video_path):
    cmd = [
        "ffprobe", "-select_streams", "v:0",
        "-show_entries", "frame=pts",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pts_list = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return pts_list

class VideoFrameMarker:
    """
    This tool supports only videos with a fixed frame rate.
    """

    # SPACE: 播放/暂停视频
    # D: 跳转到下一帧
    # A: 跳转到前一帧
    # M: 标记当前帧或取消标记
    # S: 保存标记的帧信息到文件
    # ESC: 退出程序
    def __init__(self, video_path:str, timeStamp_path):
        self.timeStamp = timeStamp_path
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]
        # self.frame_iter = self.container.decode(self.stream)
        self.total_frames = self.stream.frames
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else 0
        self.time_base = self.stream.time_base
        base, _ = os.path.splitext(video_path)
        frame_pts_temp_path = base + '_pts.json'
        if os.path.isfile(frame_pts_temp_path):
            with open(frame_pts_temp_path, 'r') as file:
                self.frame_pts = json.load(file)
        else:
            self.frame_pts = get_all_pts_ffprobe(video_path)
            with open(frame_pts_temp_path, 'w') as file:
                json.dump(self.frame_pts,file)

        if self.total_frames is not None:
            assert len(self.frame_pts) == self.total_frames
            print('frame number and index checked')
        self.container.close()
        self.container = av.open(video_path)
        self.stream = self.container.streams.video[0]        
        self.marked_frames = []
        self.marked_pts = []
        self.current_frame = 0
        self.current_pts = None
        self.paused = True
        self.window_name = 'VideoFrameMarker'

        print(f"videos: {self.total_frames} frames | {self.fps:.2f} FPS")
        print("SPACE: play/pause")
        print("D : to next frame")
        print("A : to previous frame")
        print("M : mark/unmark this frame")
        print("S : save")
        print("ESC : qiut")
        print('supports only videos with a fixed frame rate.')
    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1440, 810)
        frame_interval = 1.0 / self.fps if self.fps > 0 else 0.05
        ret=self.display_nextFrame()
        if not ret:
            print("fail to display video")
            return None
        self._updateTitle
        last_time = time.monotonic()
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  
                self.paused = not self.paused
            elif key == ord('a') or key == ord('d'):  
                self._navigate_frames(key)
            elif key == ord('m'): 
                self._mark_frame()
            elif key == ord('s'):
                self._save_marks()
            now = time.monotonic()
            
            if not self.paused and (now - last_time) >= frame_interval:
                ret = self.display_nextFrame()
                if not ret:
                    break
                last_time = now
            self._updateTitle()
        # self.cap.release()
        self._save_marks()
        cv2.destroyAllWindows()
    
    def _updateTitle(self):
        mark_status = "marked" if self.current_frame in self.marked_frames else "unmarked"
        text = f"{mark_status} frame: {self.current_frame}/{self.total_frames}, time: {self._frame_to_time(self.current_frame)}| marked frames count: {len(self.marked_frames)}"
        cv2.setWindowTitle(self.window_name,text)

    def _navigate_frames(self, key):

        if key == ord('a'):
            new_frame = max(0, self.current_frame - 1)
        else:  
            new_frame = min(self.total_frames - 1, self.current_frame + 1)
        target_pts = self.frame_pts[new_frame]
        print(f'from {self.current_frame}| pts {self.current_pts}')

        self.container.seek(target_pts, stream=self.stream, backward=True)
        frame = next(self.container.decode(self.stream))
        pts = frame.pts
        while pts < target_pts:
            frame = next(self.container.decode(self.stream))
            pts = frame.pts
        if pts != target_pts:
            print(f'fail to jump to target pts {target_pts}, jump to nearest afterward one {pts}')
        img = frame.to_ndarray(format='bgr24')
        self.current_frame = new_frame
        self.current_pts = frame.pts
        print(f'jump to {self.current_frame}| pts {self.current_pts}')
        print('--------------------------------------')
        cv2.imshow(self.window_name, img)
        self.paused = True

    def display_nextFrame(self):
        try:
            frame = next(self.container.decode(self.stream))
            img = frame.to_ndarray(format='bgr24')
            cv2.imshow(self.window_name, img)
            self.current_pts = frame.pts
            self.current_frame += 1
            return True
        except StopIteration:
            print("display done")
            return False
        except Exception as e:
            print(f"error when reading frame: {e}")
            return False

    def _mark_frame(self):
        if self.current_frame not in self.marked_frames:
            self.marked_frames.append(self.current_frame)
            self.marked_pts.append(self.current_pts)
            # self.marked_frames.sort()
            # self.marked_pts.sort()
            print(f"marked: {self.current_frame} (time: {self._frame_to_time(self.current_frame)}|pts: {self.current_pts})")
        else:
            self.marked_frames.remove(self.current_frame)
            self.marked_pts.remove(self.current_pts)
            print(f"unmarked: {self.current_frame}")
    
    def _save_marks(self):
        if len(self.marked_pts) == 0:
            print("nothing to save")
            return
        assert len(self.marked_frames) == len(self.marked_pts), f"error, frame index and frame pts number mis-match"
        self.marked_frames.sort()
        self.marked_pts.sort()
        with open(self.timeStamp, 'w') as f:
            f.write("frameID,fps,time\n")
            for frame,pts in zip(self.marked_frames,self.marked_pts):
                time_str = self._pts_to_time(pts)
                f.write(f"{frame},{pts},{time_str}\n")
        
        print(f"save {len(self.marked_frames)} marked frame time stamps to {self.timeStamp}")
    
    def _frame_to_time(self, frame_num):
        total_seconds = frame_num / self.fps
        hours = int(total_seconds // 3600)
        minutes = int(total_seconds // 60) % 60
        seconds = total_seconds % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    def _pts_to_time(self,pts):
        return str(self.time_base*pts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration from video.")
    parser.add_argument('-v',"--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument('-t', '--time_stamp_path',default='./marked_frames.csv',type=str,help="Path to the time stamp file")
    args = parser.parse_args()
    marker = VideoFrameMarker(args.video_path, args.time_stamp_path)
    marker.run()