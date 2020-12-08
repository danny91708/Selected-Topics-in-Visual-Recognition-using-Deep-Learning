# Instance segmentation
Instance segmentation on the Tiny PASCAL VOC dataset implemented with [DetectoRS](https://arxiv.org/pdf/2006.02334.pdf).
![alt text](https://github.com/danny91708/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW3/instanceSegmentation.png?raw=true)


## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-9800X CPU @ 3.80GHz
- 2x NVIDIA TITAN RTX


## Requriments
- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

The compatible MMDetection and MMCV versions are as below. Please install the correct version of MMCV to avoid installation issues.

| MMDetection version |    MMCV version     |
|:-------------------:|:-------------------:|
| master              | mmcv-full>=1.1.5, <1.3|
| 2.7.0               | mmcv-full>=1.1.5, <1.3|
| 2.6.0               | mmcv-full>=1.1.5, <1.3|
| 2.5.0               | mmcv-full>=1.1.5, <1.3|
| 2.4.0               | mmcv-full>=1.1.1, <1.3|
| 2.3.0               | mmcv-full==1.0.5|
| 2.3.0rc0            | mmcv-full>=1.0.2    |
| 2.2.1               | mmcv==0.6.2         |
| 2.2.0               | mmcv==0.6.2         |
| 2.1.0               | mmcv>=0.5.9, <=0.6.1|
| 2.0.0               | mmcv>=0.5.1, <=0.5.8|

Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.


## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions
](https://pytorch.org/), e.g.,

    ```shell
    conda install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt pacakge,
    you can use more CUDA versions such as 9.0.

3. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```

4. Clone the MMDetection repository.

    ```shell
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    ```

5. Install build requirements and then install MMDetection.

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

Note:

a. Following the above instructions, MMDetection is installed on `dev` mode
, any local modifications made to the code will take effect without the need to reinstall it.

b. If you would like to use `opencv-python-headless` instead of `opencv
-python`,
you can install it before installing MMCV.

c. Some dependencies are optional. Simply running `pip install -v -e .` will
 only install the minimum runtime requirements. To use optional dependencies like `albumentations` and `imagecorruptions` either install them manually with `pip install -r requirements/optional.txt` or specify desired extras when calling `pip` (e.g. `pip install -v -e .[optional]`). Valid keys for the extras field are: `all`, `tests`, `build`, and `optional`.


## Dataset
Download the Tiny PASCAL VOC dataset [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK).

After downloading the dataset, set the data directory into this form:
```
dataset
  ├── train_images
  |    ├── 2007_000033.jpg
  |    ├── 2007_000042.jpg
  |    └── ...
  ├── test_images
  |    ├── 2007_000629.jpg
  |    ├── 2007_001175.jpg
  |    └── ...
  ├── pascal_train.json
  └── test.json
```
Tiny VOC dataset contains only 1,349 training images, 100 test images with 20 common object classes.


## Preprocessing
The json files provided by TAs are the COCO dataset style, so they can be used directly.
Just need to 











1. Put `detectors_cascade_rcnn_r50_1x_hw3.py` into `mmdetection-master/configs/detectors/`.


## Training
To train a model from scratch, run the following command:
```
$ ./darknet detector train data/obj.data cfg/yolo-obj.cfg build/darknet/x64/yolov4.conv.137 -dont_show -map -gpus 0,1
```

The trained weights file is [here](https://drive.google.com/file/d/1YthminCrK2qNinLh7awFHOxp5hyypONo/view?usp=sharing), and the test result is mAP 0.44583.

Here is the training chart.
![alt text](https://github.com/danny91708/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW2/trainingChart.png?raw=true)
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

