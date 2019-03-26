import sys
from PyQt5.QtWidgets import QWidget, QPushButton,QHBoxLayout, QVBoxLayout, QApplication, QLabel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap

class Example(QWidget):
     
    def __init__(self):
        super().__init__()
        self.fileName = ''
        self.initUI()

    def initUI(self):
        self.lab1 = QLabel('图片', self)
        self.lab2 = QLabel('答案1', self)
        self.lab3 = QLabel('答案2', self)
        self.lab1.move(150, 200)
        self.lab1.setScaledContents(True)
        self.lab1.resize(300, 300)
        self.lab2.move(450, 200)
        self.lab3.move(450, 400)

        btn1 = QPushButton("选择图片", self)
        btn2 = QPushButton("开始识别", self)
        btn1.move(50, 50)
        btn2.move(180, 50)
        btn1.clicked.connect(self.chooseFile)           
        btn2.clicked.connect(self.beginRecognition)

         
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowTitle('车牌识别')   
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
            self.lab3.setText('请选择图片')

if __name__ == '__main__':
     
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())