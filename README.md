[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yjiC1df2)
# EV HW1: 3D Gaussian Splatting


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
conda env create -f environment.yml
conda activate ev_hw1
```

## Data Preparation
To download the dataset, run the command:
```
bash download_data.sh
```

## Training
To train the model, run the command:
```bash
python run.py
```

## Baseline
This is my Depth2PointCloud result.
![image](https://github.com/user-attachments/assets/c438e232-e9f3-4f52-8534-7bbdc34fc6f8)  
Please click the provided link below to watch the output video.  
https://youtube.com/shorts/1xXIGeVMWW0  
The loss curves and PSNR graph are shown below.  
![image](https://github.com/user-attachments/assets/ace7ee3f-d8b8-4e98-9dec-72d7788ae971)  
My depth loss didn't converge when I apply the default config.yaml, and the depth part in the render video looks yellow at all. So I adjust the parameter depth_weight. The depth loss therefore converges. The PSNR achieves at most 28.1. Total loss converges to 0.088. L1 loss converges to 0.038. Dssim loss converges to 0.041.  
## Bonus
I define a function called bonus in run.py to randomly pick samples from data as requirement mentioned. I found that PSNR increases and losses decreases as the training data decreases. But render videos looks like dropping many frames.  
The loss difference between 200 samples and 2 samples is more obvious so I offer the graph of them.  
![image](https://github.com/user-attachments/assets/cfaecad4-6010-43b1-a22c-0840e84537cf)  
This is the render video trained by 10 samples.  
https://youtube.com/shorts/R4thtgH68_Q
