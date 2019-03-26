import os
import sys
import time

from PyQt5.QtCore import Qt
from aip import AipOcr

import cv2
import numpy as np

from PIL import Image
import yaml

# gui
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QLabel
from PyQt5.QtWidgets import QFileDialog, QFrame, QMessageBox
from PyQt5.QtGui import QPixmap, QIcon


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 2 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class DrugDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    def load_shapes(self, count, height, width, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "blue")
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "total/" + filestr + "_json/info.yaml"
            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=width, height=height, mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("blue") != -1:
                # print "box"
                labels_form.append("blue")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class LPRS_DETECT(QWidget):

    def __init__(self):
        super().__init__()
        self.fileName = ''
        self.initUI()

    def initUI(self):

        self.lab = QLabel(self)
        self.lab.move(50, 30)
        self.lab.resize(800, 470)
        self.lab.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        # self.lab.setStyleSheet("QLabel{border:1px solid red;background:rgb(100,0,0,0);}}")

        self.lab1 = QLabel(self)
        self.lab2 = QLabel(self)
        self.lab3 = QLabel(self)
        self.lab4 = QLabel(self)

        self.lab1.move(80, 115)
        self.lab1.resize(350, 350)
        self.lab1.setScaledContents(True)
        self.lab1.setText("   显示图片")
        self.lab1.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lab1.setStyleSheet("QLabel{background:rgb(200,200,200,120);}"
                                "QLabel{color:rgb(1,1,1,120);font-size:20px;font-weight:bold;font-family:华文楷体;}")
        self.lab2.move(470, 115)
        self.lab2.resize(350, 225)
        self.lab2.setScaledContents(False)
        self.lab2.setAlignment(Qt.AlignCenter)
        self.lab2.setText("显示图片                                  ")
        self.lab2.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lab2.setStyleSheet("QLabel{background:rgb(200,200,200,120);}"
                                "QLabel{color:rgb(1,1,1,120);font-size:20px;font-weight:bold;font-family:华文楷体;}")
        self.lab3.move(470, 365)
        self.lab3.resize(350, 100)
        self.lab3.setText("   输出结果")
        self.lab3.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.lab3.setStyleSheet("QLabel{background:rgb(200,200,200,120);}"
                                "QLabel{color:rgb(1,1,1,120);font-size:20px;font-weight:bold;font-family:华文楷体;}")

        btn1 = QPushButton("选择图片", self)
        btn1.move(80, 50)
        btn1.resize(150, 40)
        btn1.setStyleSheet(
            "QPushButton{background:rgb(0,191,255,120);border-radius:5px;font-family:华文楷体;font-size:20px;}"
            "QPushButton:hover{color:white}"
            "QPushButton:Pressed{color:black; background:rgb(127, 255, 0,120)}")
        btn2 = QPushButton("开始识别", self)
        btn2.move(470, 50)
        btn2.resize(150, 40)
        btn2.setStyleSheet(
            "QPushButton{background:rgb(0,191,255,120);border-radius:5px;font-family:华文楷体;font-size:20px;}"
            "QPushButton:hover{color:white}"
            "QPushButton:Pressed{color:black; background:rgb(127, 255, 0,120)}")
        btn1.clicked.connect(self.chooseFile)
        btn2.clicked.connect(self.beginRecognition)

        self.setGeometry(100, 100, 900, 540)
        self.setWindowTitle('车牌识别')
        self.setWindowIcon(QIcon('logo.png'))
        self.show()

    def chooseFile(self):
        file, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./", "JPG File (*.jpg);;PNG File (*.png)")
        if file == '':
            return
        self.fileName = file
        pix = QPixmap(self.fileName)
        self.lab1.setPixmap(pix)

    def beginRecognition(self):
        if self.fileName == '':
            QMessageBox.warning(self, "警告", "\n请选择一张图片          ", QMessageBox.Ok)
            return

        start = time.time()
        result = imageProcess(self.fileName)
        during = time.time() - start

        if len(result) == 0:
            QMessageBox.warning(self, "警告", "\n未检测到车牌      ", QMessageBox.Ok)
        else:
            pix = QPixmap("carplate.jpg")
            self.lab2.setPixmap(pix)
            self.lab3.setText("运行结果："+result[0]['words']+"\n运行时间：%.3ss" % str(during))
            self.lab3.setStyleSheet("QLabel{background:rgb(200,200,200,120);}"
                                "QLabel{color:rgb(255,69, 0,160);font-size:25px;font-weight:bold;font-family:华文楷体;}")


def imageProcess(path):
    print(path)
    if path == None:
        return
    original_image = cv2.imread(path)
    dstImage = cv2.resize(original_image,(640,480),interpolation=cv2.INTER_CUBIC)
    results = model.detect([dstImage], verbose=0)
    r = results[0]
    if 0 == len(r['class_ids']):
        print("no plates found")
        return []
    if r['scores'] >= 0.9:
        y1,x1,y2,x2 = r['rois'][0]
        if abs(y1 - y2) > 15:
            cv2.imwrite("carplate.jpg",dstImage[y1:y2,x1-20:x2+20])
        else:
            cv2.imwrite("carplate.jpg",dstImage[y1-3:y2+3,x1-20:x2+20])
        # 图片转化为字节流
        converted = cv2.imencode('.png', dstImage[y1-3:y2+3,x1-20:x2+20])[1]
        #
        """ 带参数调用通用文字识别（高精度版） """
        res = client.basicAccurate(converted)
        return res['words_result']
        # if res['words_result_num'] >= 1:
        #     return res['words_result'][0]['words']
    else:
        print(r['scores'])
        return []


if __name__ == "__main__":
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    model_path = model.find_last()

    # Load trained weights
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Test on a random image
    APP_ID = '15257392'
    API_KEY = 'y7qzC7F0BjyRCRqtmjKk4Req'
    SECRET_KEY = 'QuHNLHGZKYbk7cVf6GMl655CtNN1Kc6l'
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    app = QApplication(sys.argv)
    ex = LPRS_DETECT()
    sys.exit(app.exec_())
