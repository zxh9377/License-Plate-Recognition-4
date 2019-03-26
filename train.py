#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
from PIL import Image
import yaml

from threading import Thread

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# gui
from PyQt5 import QtCore,QtGui
from PyQt5.QtWidgets import QWidget, QPushButton,QHBoxLayout, QVBoxLayout, QApplication, QLabel
from PyQt5.QtWidgets import QFileDialog, QFrame, QMessageBox, QTextEdit, QGridLayout
from PyQt5.QtGui import QIcon,QTextCursor

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

iter_num = 0


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
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1





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

    def draw_mask(self, num_obj, mask, image):
        for index in range(num_obj):
            for i in range(width):
                for j in range(height):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_shapes(self, count, height, width, jsonfiles, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "blue")

        for i in range(count):
            img_floder = dataset_root_path + jsonfiles[i] + "/img.png"
            mask_floder = dataset_root_path + jsonfiles[i] + "/label.png"
            yaml_floder = dataset_root_path + jsonfiles[i] + "/info.yaml"
            self.add_image("shapes", image_id=i, path=img_floder,
                           width=width, height=height, mask_path=mask_floder, yaml_path=yaml_floder)

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



class LPRS_TRAIN(QWidget):

    def write(self, text):
        self.text.moveCursor(QTextCursor.End)
        self.text.insertPlainText(str(text))

    def flush(self):
        pass

    def __del__(self):
        sys.stdout = sys.__stdout__

    def normalOutputWritten(self):
        pass

    def __init__(self):
        super().__init__()
        self.fileName = ''
        self.initUI()

        sys.stdout = self  #########################
        config.display()
        sys.stdout = sys.__stdout__


    def initUI(self):
        btn1 = QPushButton("选择数据文件夹", self)
        btn1.setFixedSize(150, 40)
        btn1.setStyleSheet(
            "QPushButton{background:rgb(0,191,255,120);border-radius:5px;font-family:华文楷体;font-size:20px;}"
            "QPushButton:hover{color:white}"
            "QPushButton:Pressed{color:black; background:rgb(127, 255, 0,120)}")
        btn2 = QPushButton("标注数据", self)
        btn2.setFixedSize(150, 40)
        btn2.setStyleSheet(
            "QPushButton{background:rgb(0,191,255,120);border-radius:5px;font-family:华文楷体;font-size:20px;}"
            "QPushButton:hover{color:white}"
            "QPushButton:Pressed{color:black; background:rgb(127, 255, 0,120)}")

        btn3 = QPushButton("生成训练数据", self)
        btn3.setFixedSize(150, 40)
        btn3.setStyleSheet(
            "QPushButton{background:rgb(0,191,255,120);border-radius:5px;font-family:华文楷体;font-size:20px;}"
            "QPushButton:hover{color:white}"
            "QPushButton:Pressed{color:black; background:rgb(127, 255, 0,120)}")
        btn4 = QPushButton("开始训练", self)
        btn4.setFixedSize(150, 40)
        btn4.setStyleSheet(
            "QPushButton{background:rgb(0,191,255,120);border-radius:5px;font-family:华文楷体;font-size:20px;}"
            "QPushButton:hover{color:white}"
            "QPushButton:Pressed{color:black; background:rgb(127, 255, 0,120)}")

        btn1.clicked.connect(self.chooseFile)
        btn2.clicked.connect(self.labelme)
        btn3.clicked.connect(self.generateData)
        btn4.clicked.connect(self.training)

        self.text = QTextEdit()
        self.text.setFixedSize(650, 300)
        self.text.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.text.setStyleSheet("QTextEdit{background:rgb(0,0,0,220);}"
                                "QTextEdit{color:white;font-size:18px;font-family:Times New Roman;}")

        grid = QGridLayout()
        grid.setSpacing(15)
        grid.addWidget(btn1, 0, 0)
        grid.addWidget(btn2, 0, 1)
        grid.addWidget(btn3, 1, 0)
        grid.addWidget(btn4, 1, 1)
        grid.addWidget(self.text, 2, 0, 1, 2)

        self.setLayout(grid)
        self.setGeometry(100, 100, 700, 450)
        self.setWindowTitle('车牌识别')
        self.setWindowIcon(QIcon('logo.png'))
        self.show()

    def chooseFile(self):
        open = QFileDialog()
        self.path = open.getExistingDirectory()
        sys.stdout = self  #########################
        print("您已选择数据路径：",self.path)
        sys.stdout = sys.__stdout__

        # self.path = open.getExistingDirectory()

    def labelme(self):
        os.system('labelme')

    def generateData(self):
        json_lists = []
        for i in os.listdir(self.path):
            if '.json' in i:
                json_lists.append(i)
        sys.stdout = self  #########################
        for json_file in json_lists:
            os.system('labelme_json_to_dataset ' + self.path + '/' + json_file)
            print('labelme_json_to_dataset ' + self.path + '/' + json_file)
        sys.stdout = sys.__stdout__

    def training(self):
        train_net(self.path + '/')
        sys.stdout = self  #########################
        print("finish training")
        sys.stdout = sys.__stdout__



def train_net(dataset_root_path):
    jsonfiles = []
    for i in os.listdir(dataset_root_path):
        if os.path.isdir(dataset_root_path + i):
            jsonfiles.append(i)
    # yaml_floder = dataset_root_path
    count = len(jsonfiles)
    print("data size:",count)
    # train and val set
    dataset_train = DrugDataset()
    dataset_train.load_shapes(count, height, width, jsonfiles, dataset_root_path)
    dataset_train.prepare()

    dataset_val = DrugDataset()
    dataset_val.load_shapes(count, height, width, jsonfiles, dataset_root_path)
    dataset_val.prepare()


    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # image_net, coco, or last
    init_with = "coco"

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=1,
                layers='all')


if __name__ == '__main__':

    width = 640
    height = 480
    config = ShapesConfig()
    app = QApplication(sys.argv)
    ex = LPRS_TRAIN()
    sys.exit(app.exec_())
