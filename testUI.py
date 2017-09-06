# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:43:44 2017

test UI

@author: st
"""

import sys
import numpy as np
import os
import csv
import SimpleITK as sitk
import operator
import pandas as pd 
#import sun_radiomics as srad
from radiomics import featureextractor
import collections
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import time
class testUI():
    ''' tst '''
    def __init__(self):
        '''
        Args:
            image: Input image
            tumour_mask: Binary image containing the GTV
        Returns: dict containing the features for the image
        '''
        self.ls_images = []
        # (self.image_path.text())
        self.strPath = '/media/panda/panda/1.radiomics/3.data/'
        self.params = '/media/panda/panda/0.git/pyradiomics/pyradiomics/examples/exampleSettings/exampleMR_NoResampling.yaml'
        self.csv_path = '/media/panda/panda/1.radiomics/2.codes/survival_data.csv'
        self.save_path = '/media/panda/panda/1.radiomics/2.codes/features.csv'
        lsFiles = os.listdir(self.strPath)
        # load the images
        for files in lsFiles:
           # print(files)
            self.tmpFiles = self.strPath + files

    def read_survival_csv(self):
        '''read the survival information and age of the patient'''
        self.ls_survival = []
        # read the csv file
        with open(self.csv_path) as cf:
            spreader = csv.reader(cf, delimiter=',', quotechar='\'')
            for row in spreader:
                if not 'Age' in row:  # remove the csv file title
                    self.ls_survival.append(row)
    def read_images(self):
        ls_images = []
        lsFiles = os.listdir(self.strPath)
        # load the images
        for files in lsFiles[0:1]:
            tmpFiles = self.strPath + files
            images = os.listdir(tmpFiles)
            ls_tmp = [0, 0, 0, 0, 0]
            for img in images:
                # the order : t1, t1ce, t2, flair, seg
                paths = tmpFiles + '/' + img
                if '_t1.' in img:
                    ls_tmp[0] = paths
                elif '_t1ce.' in img:
                    ls_tmp[1] = paths
                elif '_t2.' in img:
                    ls_tmp[2] = paths
                elif '_flair.' in img:
                    ls_tmp[3] = paths
                elif '_seg.' in img:
                    ls_tmp[4] = paths
            ls_images.append(ls_tmp)
        return ls_images
    def make_masks(self, image_seg):
        '''get the mask from the initial segmentation files, 1, 2, 4 stands for different regions'''
        mask4 = image_seg==4
        mask2 = image_seg==2
        mask1 = image_seg==1
        mask21 = mask2 + mask1
        mask41 = mask4 + mask1
        mask42 = mask4 + mask2
        mask421 = mask4 + mask2 + mask1
        # mask4 = sitk.BinaryThreshold(
        #     image_seg, lowerThreshold=3.9, upperThreshold=5.0)
        # mask42 = sitk.BinaryThreshold(
        #     image_seg, lowerThreshold=1.9, upperThreshold=5.0)
        # mask421 = sitk.BinaryThreshold(
        #     image_seg, lowerThreshold=0.9, upperThreshold=5.0)
        return mask1, mask2, mask4, mask21, mask41, mask42, mask421
    def normalization(self):
        # missing values
        X = np.array(self.ls_features)
        imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
        imp.fit(X)
        # normalization
        X_normalize = preprocessing.normalize(X[:,:-1], norm = 'l2')
        X_train =  np.concatenate((X[:,:-1], X_normalize), axis=1)
        Y_train = X[:,-1:]

    def save_feature(self):
        (pd.DataFrame.from_dict(data = self.dict_features, orient = 'index').to_csv(self.save_path, header=False))    
    
    def extract_one_image(self, image, mask):
        pass

    def get_one_radiomics(self, items, title = False):
        start = time.clock()
        image_t1 = sitk.ReadImage(items[0])
        image_t1ce = sitk.ReadImage(items[1])
        image_t2 = sitk.ReadImage(items[2])
        image_flair = sitk.ReadImage(items[3])
        image_seg = sitk.ReadImage(items[4])
        mask1, mask2, mask4, mask21, mask41, mask42, mask421 = self.make_masks(image_seg)
        sitk.WriteImage(image_seg, '/media/panda/panda/seg.nii.gz')
        sitk.WriteImage(mask4, '/media/panda/panda/mask4.nii.gz')
        sitk.WriteImage(mask2, '/media/panda/panda/mask2.nii.gz')
        sitk.WriteImage(mask1, '/media/panda/panda/mask1.nii.gz')
        sitk.WriteImage(mask42, '/media/panda/panda/mask42.nii.gz')
        sitk.WriteImage(mask41, '/media/panda/panda/mask41.nii.gz')
        sitk.WriteImage(mask421, '/media/panda/panda/mask421.nii.gz')
        sitk.WriteImage(mask21, '/media/panda/panda/mask21.nii.gz')
        sitk.WriteImage(image_t1, '/media/panda/panda/t1.nii.gz')
        sitk.WriteImage(image_t1ce, '/media/panda/panda/t1ce.nii.gz')
        sitk.WriteImage(image_t2, '/media/panda/panda/t2.nii.gz')
        sitk.WriteImage(image_flair, '/media/panda/panda/flair.nii.gz')
        ls_temp = []
        extractor = featureextractor.RadiomicsFeaturesExtractor(self.params)
        # print(items[0])
        # result41 = extractor.execute(image_t1, mask4)
        # print(41)
        # result42 = extractor.execute(image_t1ce, mask4)
        # print(42)
        # result43 = extractor.execute(image_t2, mask4)
        # print(43)
        # result44 = extractor.execute(image_flair, mask4)
        # print(44)
        # result421 = extractor.execute(image_t1, mask42)
        # print(421)
        # result422 = extractor.execute(image_t1ce, mask42)
        # print(422)
        # resulgt423 = extractor.execute(image_t2, mask42)
        # print(423)
        # result424 = extractor.execute(image_flair, mask42)
        # print(424)
        result4211 = extractor.execute(image_t1, mask421)
        print(4211)
        result4212 = extractor.execute(image_t1ce, mask421)
        print(4212)
        result4213 = extractor.execute(image_t2, mask421)
        print(4213)
        result4214 = extractor.execute(image_flair, mask421)
        print(4214)
        ls_temp.extend([v for v in result41.values()][7:])
        ls_temp.extend([v for v in result42.values()][7:])
        ls_temp.extend([v for v in result43.values()][7:])
        ls_temp.extend([v for v in result44.values()][7:])
        ls_temp.extend([v for v in result421.values()][7:])
        ls_temp.extend([v for v in result422.values()][7:])
        ls_temp.extend([v for v in result423.values()][7:])
        ls_temp.extend([v for v in result424.values()][7:])
        ls_temp.extend([v for v in result4211.values()][7:])
        ls_temp.extend([v for v in result4212.values()][7:])
        ls_temp.extend([v for v in result4213.values()][7:])
        ls_temp.extend([v for v in result4214.values()][7:])
        self.dict_features = dict()
        self.dict_all = dict()
        self.ls_features = []
        for keys in self.ls_survival:
            if keys[0] in items[0]:
                ls_temp.append(keys[1])
                ls_temp.append(keys[2])
#                ls_temp.append(keys[0])
                self.dict_features[keys[0]] = [np.float64(v) for v in ls_temp]
                self.ls_features.append([np.float64(v) for v in ls_temp])
        
        if title:
            ls_title = []
            ls_title.extend([v for v in result41.keys()])
            # ls_title.extend([v for v in result42.keys()][7:])
            # ls_title.extend([v for v in result43.keys()][7:])
            # ls_title.extend([v for v in result44.keys()][7:])
            # ls_title.extend([v for v in result421.keys()][7:])
            # ls_title.extend([v for v in result422.keys()][7:])
            # ls_title.extend([v for v in result423.keys()][7:])
            # ls_title.extend([v for v in result424.keys()][7:])
            # ls_title.extend([v for v in result4211.keys()][7:])
            # ls_title.extend([v for v in result4212.keys()][7:])
            # ls_title.extend([v for v in result4213.keys()][7:])
            # ls_title.extend([v for v in result4214.keys()][7:])
            self.dict_features['title'] = ls_title
        stop = time.clock()
        print('seconds time :')
        print(stop - start)

    def testRadiomics(self):
        ls_images = self.read_images()
        # calculate the image features
        n = 0
        for items in ls_images:
            if n == 0:
                self.get_one_radiomics(items, title = False)
            else:
               self.get_one_radiomics(items)
            n += 1
            print(n)
        self.save_feature()
te = testUI()
te.read_survival_csv()
te.testRadiomics()
te.normalization()
