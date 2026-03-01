from ultralytics import YOLO
from ultralytics import RTDETR
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

if __name__ == '__main__':

    model = YOLO(r'ultralytics/cfg/models/HR-SemNet/yolov8x-HR_SemNet.yaml')
    model.info()

    # model.train(
    #     cfg='ultralytics/cfg/default.yaml',
    #     data='VisDrone.yaml',
    #     task='detect',
    #     epochs=300,
    #     patience=50,
    #     batch=4,
    #     imgsz=640,
    #     optimizer='SGD',
    #     lr0=0.01,
    #     lrf=0.1,
    #     half=False,
    #     workers=8,
    #     save_json=False,
    #     cache=False,
    #     resume=False,
    #     momentum=0.937,
    #     val=True,
    # )

