# Docker

Currently, OpenCV for GPU use (DNN module) needs to be compiled manually. For this, it is good to use Docker. We chose to use the DNN module to run YOLO inference in order to eliminate the dependency on Darknet within this Docker. However, YOLOv5 started using PyTorch instead of Darknet. I tried exporting a YOLOv5 model to ONNX to use in OpenCV::DNN, but it resulted in an error. The YOLOv5 model does get exported, but when running inference in OpenCV, an error occurs. On the other hand, YOLOv5 already provides a Dockerfile that works well, making it easier to modify the Python inference code.

We have the following scenario: when running YOLOv3, we depend on OpenCV::DNN, and therefore require a complete local compilation, in which case using Docker is worthwhile. In the case of YOLOv5, inference is done directly in PyTorch. Thus, we somewhat depend on the future of YOLO. At the moment, with Joseph Redmond stepping away, Darknet has no future. Using Torch directly seems to be the way to avoid relying on a customized OpenCV. Lately, installing Torch has become easier than compiling OpenCV for GPU.

Besides all this, it is surprisingly very, very laborious to maintain a Docker setup that aligns (NVIDIA driver) -> CUDA and cuDNN -> TensorFlow -> OpenCV. It is easier to use one of the containers (docker pull) from Datamachines: https://github.com/datamachines/cuda_tensorflow_opencv

How to compile OpenCV from scratch with GPU support and DNN module:
https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/
