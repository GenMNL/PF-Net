# PF-Net
Point Flactal Network<br>
[[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf)<br>
This code is optimized for users who want to learn only one category competion.

## Environmet
- python 3.8.10
- cuda 11.3
- pytorch 1.12.1
- pytorch3d 
- open3d 0.13.0

## Dataset
PCN Dataset from [[url]](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion) or [[BaiduYun]](https://pan.baidu.com/share/init?surl=Oj-2F_eHMopLF2CWnd8T3A).(from [[PointTr Github]](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md))
You can use other datasets if it has json file same style with PCN Dataset.

## Usage
training
```python
python train.py
```
evaluation
```python
python eval.py
```
Settings for training and evaluation are in [options.py](https://github.com/GenMNL/PF-Net/blob/main/options.py).
