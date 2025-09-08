### 文件说明
- cam_ID.yaml:记录相机编号的配置文件，需要根据gopro相机序列号修正
- CLI tool
    - cam_init.py：初始化（usb控制启用与解除）
    - cam_shutter.py: 快门控制
    - handle_file.py: 提供文件功能
    - cam_watch.py: 监控相机状态

### 下载配置文件
- 媒体文件序号按照创建时间从新到旧排序
- 保存路径是相对执行位置的相对路径
```json
{
  "index": {
    "saving place 1": [media index 1, media index 2, ...],
    "saving place 1": [media index 3, ...]
  }
}
```