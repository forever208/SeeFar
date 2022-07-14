# SeeFar: Vehicle Speed Estimation and Flow Analysis from a Moving UAV

## Abstract
Visual perception from drones has been largely investigated for Intelligent Traffic Monitoring System (ITMS) recently. In this paper, we introduce SeeFar to achieve vehicle speed estimation and traffic flow analysis based on YOLOv5 and DeepSORT from a moving drone. SeeFar differs from previous works in three key ways: the speed estimation and flow analysis components are integrated into a unified framework; our method of predicting car speed has the least constraints while maintaining a high accuracy; our flow analysor is direction-aware and outlier-aware. Specifically, we design the speed estimator only using the camera imaging geometry, where the transformation between world space and image space is completed by the variable Ground Sampling Distance. Besides, previous papers do not evaluate their speed estimators at scale due to the difficulty of obtaining the ground truth, we therefore propose a simple yet efficient approach to estimate the true speeds of vehicles via the prior size of the road signs. We evaluate SeeFar on our ten videos that contain 929 vehicle samples. Experiments on these sequences demonstrate the effectiveness of SeeFar by achieving 98.0% accuracy of speed estimation and 99.1% accuracy of traffic volume prediction, respectively.

<p align="left">
  <img src="https://github.com/forever208/SeeFar/blob/master/utils/github_imgs/demo.png" width='100%' height='100%'/>
</p>

For more technical details, please refer to [our paper](https://link.springer.com/chapter/10.1007/978-3-031-06433-3_24).

## 【0】Introduction
This repo referenced [YOLOv5_DeepSORT_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).

The backbone YOLOv5 used in this project is  [v5.0 YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v5.0), you can easily upgrade to 6.0 by replacing the whole folder `models/` and `scrips/`


<!-- Running the repo in Colab is recommended, copy the file [YOLOv5_DeepSORT.ipynb](https://colab.research.google.com/drive/1AEAgVhDKNsmUmmDRG9dP9y4VToC4pVn1?usp=sharing), then run it on Colab. (remember to change the runtime type to GPU in Colab) -->




## 【1】Installation 

* Python >= 3.7
* Pytorch >= 1.7

Create a new conda environment called SeeFar, install pytorch
```
conda create --name SeeFar python=3.8
conda activate SeeFar

# for cpu
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch

# for GPU (note the cuda version of your computer)
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```

repo clone and dependencies installation 
```
git clone https://github.com/forever208/SeeFar.git

cd SeeFar/
pip install -r requirements.txt
```

## 【2】Demo
First, download our video dataset from [GoogleDrive](https://drive.google.com/file/d/1PtRJYQV-STRnRB_Cm77mC7Lj7CVtINIY/view?usp=sharing) and put into the folder `video/`

Only one video is public available right now, if you are interested in the whole dataset, please email me (ningmang666@gmail.com)

The video filename shows the parameters: 
```
10 -- video number
H45 -- the height of the drone, 45m
theta25 -- the tilt angle of the drone camera, 25 degree
20 -- the flying speed of the drone, 20 km/h
```

then download our weights file `yolov5m_best.pt` into folder `weights/`
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qIYuoKnkwnzijSfnI__HEhhfSEx0-rRn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qIYuoKnkwnzijSfnI__HEhhfSEx0-rRn" -O yolov5m_best.pt && rm -rf /tmp/cookies.txt
```


finally, run the demo script:
```
python demo.py --video video/10_H45_theta25_20km.MP4 --output video/result.mp4 --model yolov5m_best --cam_angle 25 --drone_h 45 --drone_speed 20
```


