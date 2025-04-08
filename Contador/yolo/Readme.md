# Yolo

We use both v3 and v5.
For v3, use `yoloVideoDetect.py`, which requires OpenCV compiled with GPU and the DNN module.
For v5, use `yolov5Detect.py` with the [yolov5](https://github.com/ultralytics/yolov5) environment, either in Docker or pip venv.


## Yolo V3
Command line: the input is a video, and the output is a YAML with the detections.
V3 has some specifications we use, such as the cfg with batch size 1, used for inference. In this case, `yolo_v3_1batch_anchors.cfg` available in the folder is a yolov3 configured for `batch_size = 1` for frame-by-frame video inference. Additionally, it is necessary to provide the weights file and the category names file (.names).


## Yolo V5
Command line: the input is a video, and the output is a YAML with the detections.
Example:
```
 python yolov5Detect.py --source ../../yolov5/1_2019-04-04_23-00-00.mp4 --weights ../../yolov5/weights/yolov5l_best_50.pt --conf-thres 0.1 --view-img --nosave --yml test.yml --yolov5_path ../../yolov5
```
I inserted the output in YAML format, the same used in the rest of our pipeline. You need to specify where you installed the [yolov5](https://github.com/ultralytics/yolov5) repository using the `--yolov5_path` argument, as it depends on that. If you want to export the video with detections, remove the `--nosave` flag.
In the tested case, since the detector had very few false positives, I used `--conf-thresh 0.1`.
