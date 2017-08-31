import sys
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox, QComboBox,
        QMenu, QPushButton, QRadioButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QTextEdit)

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import *

import sun_radiomics

import os
import csv
import numpy as np
import nibabel as nib
import numpy.ma as ma
import six
import SimpleITK as sitk
import operator
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

class Window(QWidget):
    '''
    initialization the window
    '''

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        # the grid is used as panel
        # there are several parts: load files, preprocessing, feature extraction, feature selection, classification
        grid = QGridLayout()
        # step.1. load the image correlated information
        grid.addWidget(self.load_files(), 0, 0)
        # step.2. preprocessing the image and mask
        grid.addWidget(self.preprocessing(), 1, 0)
        # step.3. feature extraction
        grid.addWidget(self.feature_extraction_type(), 2, 0)
        # step.4. feature selction
        grid.addWidget(self.feature_selection(), 3, 0)
        # step.5. feature classification
        grid.addWidget(self.classifier_type(), 4, 0)

        grid.addWidget(self.classifier_type(), 0, 1)

        self.setLayout(grid)
        self.setWindowTitle("Radiomics assistant system")
        self.resize(1000, 800)

    def load_files(self):
       # load files, the processing image and its lesion mask
        groupBox_feature = QGroupBox("Load Files")

        lbl1 = QLabel('Image data path:      ', self)
        lbl2 = QLabel('Mask key word:         ', self)
        lbl3 = QLabel('Lesion value in mask: ', self)
        lbl4 = QLabel('Survival data path:    ')

        self.image_path = QLineEdit()
        self.image_path.setText('../DATA/')
        hbox1 = QHBoxLayout()
        hbox1.addWidget(lbl1)
        hbox1.addWidget(self.image_path)

        self.mask_path = QLineEdit()
        self.mask_path.setText('_seg.nii.gz')
        hbox2 = QHBoxLayout()
        hbox2.addWidget(lbl2)
        hbox2.addWidget(self.mask_path)

        self.mask_type = QLineEdit()
        self.mask_type.setText('1,2,4')
        hbox3 = QHBoxLayout()
        hbox3.addWidget(lbl3)
        hbox3.addWidget(self.mask_type)

        self.survival_data = QLineEdit()
        self.survival_data.setText('../survival_data.csv')
        hbox4 = QHBoxLayout()
        hbox4.addWidget(lbl4)
        hbox4.addWidget(self.survival_data)

        load_button = QPushButton('load files')
        load_button.clicked.connect(self.load_bt_left_click)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addWidget(load_button)
        vbox.addStretch(1)
        groupBox_feature.setLayout(vbox)
        image_info = [self.image_path.text(), self.mask_path.text(), self.mask_type.text(), self.survival_data.text()]
        #print(image_info)
        return groupBox_feature

    def preprocessing(self):
        '''
        preprocessing the images
        '''
       # Create table
        groupBox_prep = QGroupBox('Preprocessing image')
        lbl_comb1 = QLabel('preprocessing type:')
        combo = QComboBox()
        combo.addItem('Check origins')
        combo.addItem('Normalization to [0-1]')
        combo.addItem('q')
        combo.addItem('a')
        combo.addItem('z')
        preprocess_button = QPushButton('preprocessing')

        preprocess_button.clicked.connect(self.prepreocess_bt_left_click)
        vbox = QVBoxLayout()
        vbox.addWidget(lbl_comb1)
        vbox.addWidget(combo)
        vbox.addWidget(preprocess_button)
        vbox.addStretch(1)
        groupBox_prep.setLayout(vbox)
        return groupBox_prep

    def feature_extraction_type(self):
        '''
        define the feature extration types for the image.
        '''
       # Create table
        groupBox_extraction = QGroupBox('Feature extraction')
        c1 = QCheckBox('First order')
        c2 = QCheckBox('GLCM')
        c3 = QCheckBox('GLRLM')
        c4 = QCheckBox('HOG')
        c5 = QCheckBox('Shape')
        c1.setChecked(True)
        c2.setChecked(True)
        c3.setChecked(True)
        c4.setChecked(True)
        c5.setChecked(True)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(c1)
        hbox1.addWidget(c2)
        hbox1.addWidget(c3)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(c4)
        hbox2.addWidget(c5)
        hbox2.addWidget(c4)

        #button for calculate features
        feature_button = QPushButton('feature extraction')
        feature_button.clicked.connect(self.feature_bt_left_click)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(feature_button)

        groupBox_extraction.setLayout(vbox)
        return groupBox_extraction

    def feature_selection(self):
        '''
        feature selection for the calculated radiomics features.
        '''
        groupBox_selection = QGroupBox("Feature selection")
        lbl1 = QLabel('Filter:')
        radio11 = QRadioButton("Naive Bayes")
        radio12 = QRadioButton("Naive Bayes")
        vbox1 = QVBoxLayout()
        vbox1.addWidget(lbl1)
        vbox1.addWidget(radio11)
        vbox1.addWidget(radio12)

        lbl2 = QLabel('Wrapper:')
        radio21 = QRadioButton("Naive Bayes")
        radio22 = QRadioButton("Naive Bayes")
        vbox2 = QVBoxLayout()
        vbox2.addWidget(lbl2)
        vbox2.addWidget(radio21)
        vbox2.addWidget(radio22)

        lbl3 = QLabel('Embedded:')
        radio31 = QRadioButton("Naive Bayes")
        radio32 = QRadioButton("Naive Bayes")
        vbox3 = QVBoxLayout()
        vbox3.addWidget(lbl3)
        vbox3.addWidget(radio31)
        vbox3.addWidget(radio32)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)

        feature_selection_button = QPushButton('feature selection')
        feature_selection_button.clicked.connect(
            self.feature_selection_bt_left_click)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(feature_selection_button)

        #hbox.addWidget(feature_selection_button)

        groupBox_selection.setLayout(vbox)
        return groupBox_selection

    def classifier_type(self):
        '''
        define the classifers for the features.
        '''
        groupBox_classifier = QGroupBox("Classifier")

        radio1 = QRadioButton("Naive Bayes")
        radio2 = QRadioButton("Support Vector Machine")
        radio3 = QRadioButton("K-Nearest Neighbor")
        radio4 = QRadioButton("Logistic Regression")
        radio5 = QRadioButton("Decision Tree")
        radio6 = QRadioButton("Neural Networks")
        radio1.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radio1)
        vbox.addWidget(radio2)
        vbox.addWidget(radio3)
        vbox.addWidget(radio4)
        vbox.addWidget(radio5)
        vbox.addWidget(radio6)
        vbox.addStretch(1)
        groupBox_classifier.setLayout(vbox)
        return groupBox_classifier

    def load_bt_left_click(self):
        '''
        Args:
            image: Input image
            tumour_mask: Binary image containing the GTV
        Returns: dict containing the features for the image
        '''
        image_info = [self.image_path.text(), self.mask_path.text(), self.mask_type.text(), self.survival_data.text()]
        
        lsFiles = os.listdir(self.image_path.text())
        for files in lsFiles:
            tmpFiles = self.image_path.text() + files
            images = os.listdir(tmpFiles)
            for img in images:
                img_load = sitk.ReadImage(img)
                print(img_load.GetOrigin)

        print(len(lsFiles))
    #    images = sitk.ReadImage()
    #    img = np.array()



    def prepreocess_bt_left_click(self):
        sender = self.sender()

    def feature_bt_left_click(self):
        sender = self.sender()
  
    def feature_selection_bt_left_click(self):
        sender = self.sender()



 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())
