# yolov5-deepsort
traffic flow analysis by DeepSORT, including object counting, speed estimation and dynamic area identification (main road)

 

## 【0】Introduction
This repo referenced [YOLOv5_DeepSORT_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
The backbone YOLOv5 used in this project is  [v5.0 YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v5.0)


Running the repo in Colab is recommended, copy the file [YOLOv5_DeepSORT.ipynb](https://colab.research.google.com/drive/1AEAgVhDKNsmUmmDRG9dP9y4VToC4pVn1?usp=sharing), then run it on Colab. (remember to change the runtime type to GPU in Colab)




## 【1】Installation 

* Python >= 3.7
* Pytorch >= 1.7

Create a new conda environment called DeepSORT , install pytorch-1.7.0 (Colab user can skip this step) 
```
conda create --name DeepSORT python=3.7
conda activate DeepSORT

# for cpu
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cpuonly -c pytorch

# for GPU (note the cuda version of your computer)
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

```

repo clone and dependencies installation 
```
git clone -b master https://github.com/forever208/yolov5-deepsort.git

cd yolov5-deepsort/
pip install -r requirements.txt

```

