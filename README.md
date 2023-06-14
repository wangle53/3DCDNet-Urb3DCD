# 3DCDNet-Urb3DCD
The experimental code of 3DCDNet on Urb3DCD dataset.
![image](https://github.com/wangle53/3DCDNet-Urb3DCD/assets/79884379/f7485378-86f1-49aa-ace4-a943528b2d86)
![image](https://github.com/wangle53/3DCDNet-Urb3DCD/assets/79884379/2fd36523-598b-4746-8fef-a31f398097c4)
## Requirement
```
python 3.7.4
torch 1.8.10
visdom 0.1.8.9
torchvision 0.9.0
```
## Urb3DCD Dataset
The Urb3DCD Dataset can be found from ( I. de Gélis, S. Lefèvre, and T. Corpetti, “Change Detection in Urban Point Clouds: An Experimental Comparison with Simulated 3D Datasets,” Remote Sensing, vol. 13, no. 13, p. 2629, Jul. 2021, doi: 10.3390/rs13132629.)
## Pretrained Model
The pretrained model fro Urb3DCD is available at  [[Google Drive]](https://drive.google.com/drive/folders/1ehQbfsGvOv4syc98r5PlhJDV88Q3bQlg?usp=sharing) and [[Baiduyun]](https://pan.baidu.com/s/1IUy8WFIggkIsHNyR8rTG-w) (the password is: qjmf).
## Test
Before test, please download datasets and pretrained models. Change path to your data path in configs.py. Copy pretrained models to folder './outputs_{sub_dataset_names}/best_weights', and run the following command: 
```
cd 3DCDNet_ROOT
python test.py
```
## Training
Before training, please download datasets and revise dataset path in configs.py to your path.
```
cd 3DCDNet_ROOT
python -m visdom.server
python train.py
```
To display training processing, open 'http://localhost:8097' in your browser.
## Citing 3DCDNet
If you use this repository or would like to refer the paper, please use the following BibTex entry.
## More
[My personal google web](https://scholar.google.com/citations?user=qdkY0jcAAAAJ&hl=zh-TW)
