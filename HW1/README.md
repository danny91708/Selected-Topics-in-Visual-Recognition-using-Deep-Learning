# Car brand classification
The homework website for [Kaggle](https://www.kaggle.com/c/cs-t0828-2020-hw1).

There are two streams in my architecture: Classification Stream and Attention Stream.
Classification Stream just classifies the images by ResNet-50, while Attention Stream will focus on the discriminative regions and then classifies the images.
The CMAB module is a great attention in recent years, so I use the CBAM module to produce the attention map.
I also use the Spatial Keeping module to keep the spatial information.
![alt text](https://github.com/danny91708/Selected-Topics-in-Visual-Recognition-using-Deep-Learning/blob/main/HW1/architecture.png?raw=true)

## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz x 20
- 3x NVIDIA GeForce RTX 2080 Ti

## Requriments
- python 3.6
- PyTorch 1.4.0
- torchvision 0.5.0

## Dataset
Download the car dataset [here](https://www.kaggle.com/c/cs-t0828-2020-hw1/data).

After downloading the dataset, the data directory is structured as:
```
dataset
  ├── training_data
  |    └── training_data
  |         ├── 000001.jpg
  |         ├── 000002.jpg
  |         └── ...
  ├── testing_data
  |    └── testing_data
  |         ├── 000001.jpg
  |         ├── 000002.jpg
  |         └── ...
  └── training_labels.csv
```

## Pretrained model
ResNet-50 is pretrained on ImageNet.

You can download the pretrained model [here](https://drive.google.com/file/d/1RZSCxmEbZQrAajbt871x8rvbrkfyta_v/view?usp=sharing).

## Training
To train a model from scratch, run the following command:
```
$ python main.py
```

## Make Submission
The predictions for all the epoches are in the folder `predict_csv` after finishing the training.
```
predict_csv
  ├── predict_epoch_1.csv
  ├── predict_epoch_1_combined.csv
  ├── predict_epoch_2.csv
  ├── predict_epoch_2_combined.csv
  └── ...
```

## Reference
- https://etd.lib.nctu.edu.tw/cgi-bin/gs32/tugsweb.cgi?o=dnctucdr&s=id=%22GT070756044%22.&searchmode=basic
- https://github.com/asdf2kr/BAM-CBAM-pytorch/blob/master/Models/attention.py
