# SpindlesTracker

SpindleTracker is a automatically pipline for spindles detection, tracking and analysis. 

<img src="docs/assets/workflow.png" />


## 1. Visualization demo

### Tracking video

<img src="docs/assets/demo.gif" />

### Skeletonize interface

<img src="docs/assets/interface.png" />

## 2. Install
```bash
cd SpindleTracker
pip install -r requirements.txt
pip install -e .
```

## 3. Usage
### Optimal: YOLOX-SP Training
* Refer to our another repository: [lizhogn/YOLOX-SP](https://github.com/lizhogn/YOLOX-SP), you can train it on your own dataset.

### 1. Download the demo data and YOLOX-SP ONNX model
* Demo file download: [Google Drive](https://drive.google.com/drive/folders/1C_d2gVMFe43_rwdn6I7tvl8x0cdXtRjb?usp=share_link)
(put the data to `demo/data`)

* YOLOX-SP model download: [Google Drive](https://drive.google.com/file/d/1jV5lB8FFp0J5lwaogQAP657D74x03R_K/view?usp=share_link)
(put the weight to `module/detection/weight`)

### 2. Images Detection Demo

* step2: run the following:
```bash
python demo_img.py --images path/to/input/image.png --model path/to/onnx/model
```
The results is saved in the image path (detection, mask and skeleton as shown in following).

<img src="docs/assets/image_pred.png" />


### 3.2 TIF Input Demo
* step1: import the ONNX model from YOLOX-SP and put it into `module/detection/weight` path (default path).

* step2: run the following:
```bash
python demo_tif.py --images path/to/input/image.tif --model path/to/onnx/model --save_path path/to/save
```
The results is saved in save path. NOTE that the tif need to be double channel (red and green). The other input tif need to change the image propressing file and retrain the YOLOX-SP model.

### 3.3 GUI Interface Demo
The interface demo was built with `gradio`, first run the following:
```
python app/app.py
```
then, click the localhost link. Some images were listed in the example block, you can upload your own img.
