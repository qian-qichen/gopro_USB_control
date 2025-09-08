import requests
import concurrent.futures
import threading
import time
import yaml
import argparse
import json
import os
import tqdm
import itertools
TIME_OUT = 10
CHUNK_SIZE = 8 * 1024 * 1024
HANDLE_CHAPTER_FILE = True
def read_camera_list(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    camera_list = data.get('cameras', [])
    return camera_list

def synchronize_file(ipPort,save_dir):
    response = requests.request("GET", f'http://{ipPort}/gopro/media/list')
    response.raise_for_status()
    geted = response.text
    is_json = True
    try:
        json_obj=json.loads(geted)
        order_medias = [] # reverse creaation order
        for media in json_obj['media']:
            media['fs'] = sorted(media['fs'], key=lambda x:int(x['cre']), reverse=True)
            order_medias.append(media)
        json_obj['media'] = order_medias
    except json.JSONDecodeError:
        is_json = False
        print(f"Warning: Response from {ipPort} is not valid JSON.")

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{ipPort.replace(':', '_')}_media_list.json")
    with open(filename, 'w', encoding='utf-8') as f:
        if is_json:
            json.dump(json_obj, f,indent=4) #type:ignore
        else:
            f.write(geted)
    return f"get {'valid' if is_json else 'invalid'} meida list for {ipPort}"

## not supported for gopro hero 10 black
def get_last_file_path(ipPort):
    response = requests.request("GET", f'http://{ipPort}/gopro/media/last_captured')
    print(f'http://{ipPort}/gopro/media/last_captured')
    response.raise_for_status()
    geted = response.text
    if response.status_code == 204:
        print(f"not last captured media to report for {ipPort}")
        return ipPort,None
    json_obj = False
    try:
        json_obj=json.loads(geted)
    except json.JSONDecodeError:
        print(f"Warning: Response from {ipPort} is not valid JSON.")
    return ipPort,json_obj # json_obj {'file':..., 'folder':...}

def ipPort2id(ipPort:str):
    parts = ipPort.split('.')
    return parts[1][1:]+parts[2][1:]

def get_metaData(ipPort, dir, name):
    url = f"http://{ipPort}/gopro/media/info?path={dir}/{name}"
    # querystring = {"path":f"{dir}/{name}"}
    response = requests.request("GET", url)
    response.raise_for_status()
    # print(response.headers.get('Content-Length'))
    if response.headers.get('Content-Length') == '0':
        return None
    try:
        meta_data = json.loads(response.text)
        if meta_data['s'] == '0':
            return None
        else:
            return meta_data
    except Exception as e:
        print(f"Error getting metadata: {e}")
        return None
'''
curl --request GET --url 'http://172.28.159.51:8080/gopro/media/info?path=100GOPRO/GOPR0002.JPG'
'''
def is_chaptered_video_not_first(name:str):
    id = name[2:4]
    if not id.isdigit():
        return False
    if id =='01':
        return False
    return True
def get_video_segments(ipPort, dir, name):
    """
    Given a video file name, determine if it is a chaptered video.
    If so, return a list of all chapter file names.
    Otherwise, return a list with the single file name.
    """
    meta_data = get_metaData(ipPort, dir, name)
    if meta_data is None:
        raise ValueError(f"fail to find meta data for {dir}/{name} at {ipPort}")
    if meta_data.get('ct') != '2':
        return [name]
    tail = name[4:]
    head = name[:2]
    segments = []
    for i in range(1, 100):
        chapter_name = f"{head}{i:02d}{tail}"
        meta = get_metaData(ipPort, dir, chapter_name)
        if meta is not None:
            segments.append(chapter_name)
        else:
            break
    return segments
def download_file(id, saving_dir, ipPort, dir, name, chunk_size=CHUNK_SIZE,position=0,disc_tail=""):
    saving_path = os.path.join(saving_dir, f"{id}_{dir}_{name}")
    print(f'downloading: {id}_{dir}_{name} .. ')
    try:
        response = requests.request("GET", f'http://{ipPort}/videos/DCIM/{dir}/{name}',stream=True)
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0))
        with open(saving_path, 'wb') as f,tqdm.tqdm(
            total=total, unit='B', unit_scale=True, desc=f"{id}_{dir}_{name}|{disc_tail}", ncols=80, position=position, leave=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return {"for":f"{ipPort}-{dir}/{name}","status": "download success", "path": saving_path}
    except Exception as e:
        return {"for":f"{ipPort}-{dir}/{name}","status": "download error", "error": str(e), "path": saving_path}

def download_file_chapteredVideoHandled(id, saving_dir, ipPort, dir, name:str,chunk_size=CHUNK_SIZE,position=0):

    print(f'examing {dir}/{name} from {id}:')
    to_download = get_video_segments(ipPort,dir,name)
    how_many_to_download = len(to_download)
    if how_many_to_download == 1:
        print('media in one file')
        state = download_file(id, saving_dir, ipPort, dir, name,chunk_size,position)
        return state
    else:
        print('chaptered video find')
        down_load_info = {}
        for i,n in enumerate(to_download):
            tail = f"{i+1}/{how_many_to_download}"
            state = download_file(id, saving_dir, ipPort, dir, n,chunk_size,position,disc_tail=tail)
            down_load_info[state['for']] = state['status']
        return down_load_info    
    
def delete_file(ipPort,dir, name):
    print(f'deleting: {ipPort}_{dir}_{name} ...')
    url = f"http://{ipPort}/gopro/media/delete/file?path={dir}/{name}"
    response = requests.request("GET", url)

    if response.status_code == 400:
        return  {"for":f"{ipPort}-{dir}/{name}","status": "delete failed"}
    
    return {"for":f"{ipPort}-{dir}/{name}","status": "delete sucess"}

def delete_file_chapteredVideoHandled(ipPort,dir, name):
    print(f'examing {dir}/{name} from {id}:')
    to_delete = get_video_segments(ipPort,dir,name)
    if len(to_delete) == 1:
        print('media in one file')
        state = delete_file(ipPort, dir, name)
        return state
    else:
        print('chaptered video found')
        delete_load_info = {}
        for n in to_delete:
            state = delete_file(ipPort, dir, n)
            delete_load_info[state['for']] = state['status']
        return delete_load_info

def download_file_entry(id, saving_dir, ipPort, dir, name,chunk_size=CHUNK_SIZE,chapteredVideoHandled=HANDLE_CHAPTER_FILE,posion=0):
    outs=None
    if isinstance(dir,str) and isinstance(name,str) and isinstance(saving_dir,str):
        if chapteredVideoHandled:
            outs = download_file_chapteredVideoHandled(id, saving_dir, ipPort, dir, name,chunk_size,posion)
        else:
            outs = download_file(id, saving_dir, ipPort, dir, name, chunk_size,posion)
    elif isinstance(dir,list) and isinstance(name,list) and isinstance(saving_dir, list):
        outs = []
        if chapteredVideoHandled:
            for dir,name,saving in zip(dir,name,saving_dir):
                out=download_file_chapteredVideoHandled(id, saving, ipPort, dir, name,chunk_size,posion)
                outs.append(out)
        else:
            for dir,name,saving in zip(dir,name,saving_dir):
                out = download_file(id, saving, ipPort, dir, name, chunk_size,posion)
                outs.append(out)
    return outs
def delete_file_entry(ipPort, dir, name, chapteredVideoHandled=HANDLE_CHAPTER_FILE):
    outs=None
    if isinstance(dir, str) and isinstance(name, str):
        if chapteredVideoHandled:
            outs = delete_file_chapteredVideoHandled(ipPort, dir, name)
        else:
            outs = delete_file(ipPort, dir, name)
    elif isinstance(dir, list) and isinstance(name, list):
        outs = []
        if chapteredVideoHandled:
            for d, n in zip(dir, name):
                out = delete_file_chapteredVideoHandled(ipPort, d, n)
                outs.append(out)
        else:
            for d, n in zip(dir, name):
                out = delete_file(ipPort, d, n)
                outs.append(out)        
    return outs

def find_nth_file_chapterHandled(media_json, folder, counter):
    for media in media_json.get('media', []):
        if media.get('d', '') == folder:
            count = counter
            index = 0
            files = media.get('fs', [])
            maxIndex = len(files) - 1
            while count >= 0 and index <= maxIndex:
                if not is_chaptered_video_not_first(files[index]['n']):
                    count -= 1
                index += 1
            if index > maxIndex:
                return None
            if count < 0:
                index -= 1
                return {'folder': media.get('d', ''), 'file': files[index]['n']}
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="handling media file")
    parser.add_argument('yaml_file', nargs='?', default='cam_ID.yaml', help='Path to the camera ID YAML file')
    parser.add_argument('-s','--synchronize', action='store_true', help='get a frash copy of the list of medias of the cameras, before doing other thing.')
    parser.add_argument('-mp','--media_record_path', required=False, default='./media_record',help='place to place records od medias on cameras')
    parser.add_argument('-d','--download', required=False,default=None, help="to download a file? Given number i, download the i-th newly created file, or given file name, or given the download-plan-file path")
    parser.add_argument('--delete',required=False,default=None, help="to delete a file? Given number i, delete the i-th newly created file, or given file name")
    parser.add_argument('-c','--camera_id',required=False,nargs='+',default=None,help="Specify camera IDs to overwrite those in the config file. For download/delete by file name and folder, only the first camera_id is used.")
    parser.add_argument('-f','--folder',required=False,default='100GOPRO', help="only file under this folder will count when downloading file by the reverse creation order, or combined with file name provided by --download(-d)")
    parser.add_argument('-dp','--download_path', required=False, default='./download', help='the path to store the file downloaded,will be ignored if download with a ')
    parser.add_argument('-ck','--chunk_size',required=False,default=None,type=int,help="determine chunk size while stream downloading")
    parser.add_argument('-cvho','--chaptered_video_Handling_off',required=False,action='store_true', help='Disable automatic handling of chaptered videos during download')
    # parser.add_argument('-check','--check_redownload',action='store_true', help='check error file in download_path and redownload thems')
    args = parser.parse_args()
    yaml_file = args.yaml_file
    default_camerasID = read_camera_list(yaml_file)

    camerasID = args.camera_id if args.camera_id is not None else default_camerasID
    cameras =  {id: {'ipPort':f'172.2{id[0]}.1{id[1:]}.51:8080'} for id in camerasID}
    chunk_size = args.chunk_size if args.chunk_size is not None else CHUNK_SIZE
    HANDLE_CHAPTER_FILE = not args.chaptered_video_Handling_off
    if args.synchronize:
        saveDir = args.media_record_path
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cameras)) as executor:
            futures = [executor.submit(synchronize_file, cam['ipPort'], saveDir) for cam in cameras.values()]
            for future in concurrent.futures.as_completed(futures):
                print(future.result())
    download_flag = args.download is not None
    delete_flag = args.delete is not None
    assert (args.download is None) or (args.delete is None), "please do NOT downloading and deleting at the same time"
    indentify_file = (args.download is not None) or (args.delete is not None)
    file_clue = args.download if args.download is not None else args.delete
    if indentify_file:
        target_file = {}
        if file_clue.isdigit():# 指定下载序号
            count = int(file_clue)
            print('finding the {}-th file in each camera...')
            for camID, cam in tqdm.tqdm(cameras.items()):
                media_list_path = os.path.join(args.media_record_path, f"{cam['ipPort'].replace(':', '_')}_media_list.json")
                if os.path.exists(media_list_path):
                    with open(media_list_path, 'r', encoding='utf-8') as f:
                        try:
                            media_json = json.load(f)
                            found = False
                            if HANDLE_CHAPTER_FILE:
                                result = find_nth_file_chapterHandled(media_json, args.folder, count)
                                if result is not None:
                                    target_file[cam['ipPort']] = [result]
                                    found = True
                            else:
                                for media in media_json.get('media', []):
                                    if media.get('d', '') == args.folder:
                                        files = media.get('fs', [])
                                        if len(files) > count:
                                            target_file[cam['ipPort']] = [{'folder': media.get('d', ''), 'file': files[count]['n']}]
                                            found = True
                                            break
                            if not found:
                                print(f"fail to find the {count}-th file in folder {args.folder} at {cam['ipPort']}.")
                        except Exception as e:
                            print(f"Error loading media list for {cam['ipPort']}: {e}")
                else:
                    print(f"Media list file not found for {cam['ipPort']}: {media_list_path}")

        elif isinstance(file_clue, str) and os.path.exists(file_clue): #通过文件指定多个下载序号或者格式化后的下载文件名及对应的保存位置
            with open(file_clue, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                to_download_index = obj.get('index',[])
                to_downlaod_file_name = obj.get('file_name',[])
            if len(to_download_index) != 0:
                for camID, cam in cameras.items():
                    media_list_path = os.path.join(args.media_record_path, f"{cam['ipPort'].replace(':', '_')}_media_list.json")
                    if os.path.exists(media_list_path):
                        with open(media_list_path, 'r', encoding='utf-8') as f:
                            try:
                                media_json = json.load(f)
                                found = False
                                target_file[cam['ipPort']] = []
                                for save_place, counts in to_download_index.items():
                                    for count in counts:
                                        if HANDLE_CHAPTER_FILE:
                                            result = find_nth_file_chapterHandled(media_json, args.folder, count)
                                            if result is not None:
                                                result['save_place'] = save_place
                                                target_file[cam['ipPort']].append(result)
                                                found = True
                                        else:
                                            for media in media_json.get('media', []):
                                                if media.get('d', '') == args.folder:
                                                    files = media.get('fs', [])
                                                    if len(files) > count:
                                                        target_file[cam['ipPort']].append({'folder': media.get('d', ''), 'file': files[count]['n'],'save_place':save_place})
                                                        found = True
                                                        break
                                        if not found:
                                            print(f"fail to find the {count}-th file in folder {args.folder} at {cam['ipPort']}.")
                            except Exception as e:
                                print(f"Error loading media list for {cam['ipPort']}: {e}")
                    else:
                        print(f"Media list file not found for {cam['ipPort']}: {media_list_path}")
            if len(to_downlaod_file_name) != 0:
                for save_place, fileNames in to_downlaod_file_name.items():
                    for fileName in fileNames:
                        # 解析{id}_{dir}_{name}形式的fileName并加入到target_file中
                        # 假设fileName格式为: {id}_{dir}_{name}
                        try:
                            parts = fileName.split('_', 2)
                            if len(parts) == 3:
                                cam_id, folder, file_name = parts
                                ipPort = f'172.2{cam_id[0]}.1{cam_id[1:]}.51:8080'
                                if ipPort not in target_file:
                                    target_file[ipPort] = []
                                target_file[ipPort].append({'folder': folder, 'file': file_name,'handle_chaptered_file':False,'save_place':save_place})
                            else:
                                print(f"Invalid file name format: {fileName}")
                        except Exception as e:
                            print(f"Error parsing file name {fileName}: {e}")
            
        elif args.camera_id and isinstance(file_clue, str): # 指定名称与相机
            assert args.camera_id,"need camera id for download a specific file on a specific camera"
            camID = args.camera_id[0]
            ipPort = f'172.2{camID[0]}.1{camID[1:]}.51:8080'
            folder = args.folder
            file_name = file_clue
            target_file[ipPort] = {'folder': folder, 'file': file_name}
        
        save_place_to_check = set()
        for ipPort, file_objs in target_file.items():
            if len(file_objs) == 1:
                obj = file_objs[0]
                sp = obj.get('save_place',args.download_path)
                save_place_to_check.add(sp)
                new_obj = {
                    'folder':obj['folder'],
                    'file':obj['file'],
                    'handle_chaptered_file': obj.get('handle_chaptered_file', HANDLE_CHAPTER_FILE),
                    'save_place': sp
                }
            else:
                folders = []
                files = []
                handle_flags = []
                save_place = []
                for obj in file_objs:
                    folders.append(obj['folder'])
                    files.append(obj['file'])
                    handle_flags.append(obj.get('handle_chaptered_file', HANDLE_CHAPTER_FILE))
                    sp = obj.get('save_place', args.download_path)
                    save_place_to_check.add(sp)
                    save_place.append(sp)
                new_obj = {
                    'folder':folders,
                    'file':files,   
                    'handle_chaptered_file': handle_flags,
                    'save_place':save_place                  
                }
            target_file[ipPort] = new_obj

        for path in save_place_to_check:
            os.makedirs(path,exist_ok=True)

        if download_flag:
            position_counter = itertools.count()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_file)) as executor:
                futures = [executor.submit(download_file_entry, ipPort2id(ipPort), down_task['save_place'],ipPort,down_task['folder'],down_task['file'],chunk_size, down_task['handle_chaptered_file'],next(position_counter)) for ipPort, down_task in target_file.items() ]
                for future in concurrent.futures.as_completed(futures):
                    print(future.result())
        if delete_flag:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_file)) as executor:
                futures = [
                    executor.submit(delete_file_entry, ipPort, path['folder'], path['file'], HANDLE_CHAPTER_FILE)
                    for ipPort, path in target_file.items()
                ]
                for future in concurrent.futures.as_completed(futures):
                    print(future.result())       