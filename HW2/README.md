# Street View House Numbers Detection
Object detection on the street view house numbers dataset implemented with Yolo v4.
![alt text](https://github.com/danny91708/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW2/NumbersDetection.png?raw=true)

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-9800X CPU @ 3.80GHz
- 2x NVIDIA TITAN RTX

## Requriments
* Linux
* **CMake >= 3.12**: https://cmake.org/download/
* **CUDA >= 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**

* python 3.6

## Git clone Yolo v4
```
$ git clone https://github.com/AlexeyAB/darknet
```

## Compile
### How to compile on Linux/macOS (using `CMake`)

The `CMakeLists.txt` will attempt to find installed optional dependencies like CUDA, cudnn, ZED and build against those. It will also create a shared object library file to use `darknet` for code development.

Open a shell terminal inside the cloned repository and launch:

```bash
./build.sh
```

### How to compile on Linux (using `make`)

Just do `make` in the darknet directory. (You can try to compile and run it on Google Colab in cloud [link](https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE) (press «Open in Playground» button at the top-left corner) and watch the video [link](https://www.youtube.com/watch?v=mKAEGSxwOAY) )
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights test.mp4`
* `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights zed_camera`
* You also need to specify for which graphics card the code is generated. This is done by setting `ARCH=`. If you use a never version than CUDA 11 you further need to edit line 20 from Makefile and remove `-gencode arch=compute_30,code=sm_30 \` as Kepler GPU support was dropped in CUDA 11. You can also drop the general `ARCH=` and just uncomment `ARCH=` for your graphics card.

To run Darknet on Linux use examples from this article, just use `./darknet` instead of `darknet.exe`, i.e. use this command: `./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -ext_output -dont_show -out result.json < data/test_obj.txt`


## Dataset
Download the Street View House Numbers dataset [here](https://drive.google.com/drive/u/1/folders/1Ob5oT9Lcmz7g5mVOcYH3QugA7tV3WsSl).

After downloading the dataset, set the data directory into this form:
```
darknet-master
  ├── data
       ├── obj
            ├── train
            |    ├── digitStruct.mat
            |    ├── 1.png
            |    ├── 2.png
            |    └── ...
            └── test
                 ├── 1.png
                 ├── 2.png
                 └── ...
```

## Preprocessing
0. Download the pretrained weights [here](https://drive.google.com/file/d/1_QpkXEbhclqzjDgtYtqe7GJUta7ZKHi0/view?usp=sharing), anf put `yolov4.conv.137` into `darknet-master/build/darknet/x64`.
1. Put `obj.data` and `obj.name` into `darknet-master/data`.
2. Put `yolo-obj.cfg` into `darknet-master/cfg`.
3. To make yolov4 get the annotations of each image, we convert `digitStruct.mat` into `.txt` file for each image, and produce `train_obj.txt` and `test_obj.txt` storing the path of training and test images by running the following command:
```
$ python annotation_converter.py 
```


## Training
To train a model from scratch, run the following command:
```
$ ./darknet detector train data/obj.data cfg/yolo-obj.cfg build/darknet/x64/yolov4.conv.137 -dont_show -map -gpus 0,1
```

The trained weights file is [here](https://drive.google.com/file/d/1YthminCrK2qNinLh7awFHOxp5hyypONo/view?usp=sharing), and the test result is mAP 0.44583.
To know more training details, please check [Yolo v4](https://github.com/AlexeyAB/darknet).

## Test
To test a model, choose the weights in the folder `backup`, and run the following command:
```
$ ./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_final.weights -ext_output -dont_show -out result.json < data/test_obj.txt
```

The result will be stored in `result.json`.

## Make Submission
To satisfy the rule of submission, convert `result.json` to the specific format by running the following command:
```
$ python json_converter.py
```

It will produce a new json file for submission.

## Reference
- https://github.com/AlexeyAB/darknet

