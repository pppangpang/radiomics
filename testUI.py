# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:43:44 2017

test UI

@author: st
"""

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

import sun_radiomics as srad

class testt():
    def __init__(self):
        '''
        Args:
            image: Input image
            tumour_mask: Binary image containing the GTV
        Returns: dict containing the features for the image
        '''
        self.ls_images = []
        strPath = '../DATA/'  #(self.image_path.text())
        lsFiles = os.listdir(strPath)
        # load the images
        for files in lsFiles:
           # print(files)
            self.tmpFiles = strPath + files
            
    def test(self):
        ls_images = []
        strPath = '../DATA/'  #(self.image_path.text())
        lsFiles = os.listdir(strPath)
        # load the images
        for files in lsFiles[0:5]:
            print(files)
            tmpFiles = strPath + files
            images = os.listdir(tmpFiles)
            ls_tmp = []
            tmp = []
            for img in images:
                '''the order : flair, seg, t1, t1ce, t2'''
                tmp.append(img)
                
                paths = tmpFiles + '/' + img
                image_info = sitk.ReadImage(paths)
                ls_tmp.append(image_info)
            ls_images.append(ls_tmp)
        # calculate the image features
        for items in ls_images:
            image = items[0]
            tumour_mask = items[1]
            img2 = np.array(image)
            img2 -= np.amin(image)
            img2, mask2 = srad.clip_to_bounding_box(img2, tumour_mask)
            img2 *= mask2 > 0
            features = {}
            features.update(srad.group1_features(img2[mask2 > 0]))
            features.update(srad.tumour_features(mask2, [3, 1, 1]))
            features.update(srad.gray_level_runlength_features(img2, mask2))
            features.update(srad.gray_level_cooccurrence_features(img2, mask2))
            features.update(srad.wavelet_features(img2, mask2))
            print(features)

        pass

te = testt()
te.test()


