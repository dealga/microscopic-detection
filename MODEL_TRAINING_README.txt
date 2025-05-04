->clone the yolo v11 model from ultralytics github

->before starting the training of the model, 

We must create a .yaml file inside the directory ultralytics/ultralytics/cfg/datasets/{{name of our yaml}}.yaml


->the content of the yaml should be as follows

path: /path to/datasets
train: /path to/datasets/images/train
val: /path to/datasets/images/val
test: /path to/datasets/images/test
names: ['mitotic', 'non-mitotic']
nc: 2

-> then run 

!yolo task=detect mode=train model=yolo11x.pt data=/path to/{{name of our yaml}}.yaml epochs=100 imgsz=256