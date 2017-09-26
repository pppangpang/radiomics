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
        self.strPath = '/media/panda/panda/1.radiomics/3.data/1/'
        self.params = '/media/panda/panda/0.git/pyradiomics/pyradiomics/examples/exampleSettings/exampleMR_NoResampling.yaml'
        self.csv_path = '/media/panda/panda/1.radiomics/2.codes/survival_data.csv'
        self.save_path = '/media/panda/panda/0.git/radiomics/features/'
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
        for files in lsFiles: #[0:1]
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
        '''ds'''
        # missing values
        X = np.array(self.ls_features)
        imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
        imp.fit(X)
        # normalization
        X_normalize = preprocessing.normalize(X[:,:-1], norm = 'l2')
        X_train =  np.concatenate((X[:,:-1], X_normalize), axis=1)
        Y_train = X[:,-1:]
        pa = self.save_path + 'normalized.csv'
        df = pd.DataFrame(X_train)
        df.to_csv(pa, header=False)   
    def save_feature(self):
        '''save files'''
        pa = self.save_path + 'features.csv'
        (pd.DataFrame.from_dict(data = self.dict_features, orient = 'index').to_csv(pa, header=False))        
    def extract_one_image(self, items, mask):
        '''extract the features'''
        image_t1 = sitk.ReadImage(items[0])
        image_t1ce = sitk.ReadImage(items[1])
        image_t2 = sitk.ReadImage(items[2])
        image_flair = sitk.ReadImage(items[3])
        ls_tmp_feature = []
        extractor = featureextractor.RadiomicsFeaturesExtractor(self.params)
        result1 = extractor.execute(image_t1, mask)
        result2 = extractor.execute(image_t1ce, mask)
        result3 = extractor.execute(image_t2, mask)
        result4 = extractor.execute(image_flair, mask)
        ls_tmp_feature.extend([v for v in result1.values()][7:])
        ls_tmp_feature.extend([v for v in result2.values()][7:])
        ls_tmp_feature.extend([v for v in result3.values()][7:])
        ls_tmp_feature.extend([v for v in result4.values()][7:])
        return ls_tmp_feature
    def is_feature_calculated(self, items):
        for keys in self.ls_survival:
            if keys[0] in items[0]:
                pa = self.save_path + keys[0] + '.csv'
                return os.path.exists(pa)

    
    def save_one_feature(self, ls_feature, filename):
        pa = self.save_path + filename + '.csv'
        df = pd.DataFrame(ls_feature)
        df.to_csv(pa, header=False)

    def get_one_radiomics(self, items, title = False):
        start = time.clock()
        print(items[0])
        image_seg = sitk.ReadImage(items[4])
        mask1, mask2, mask4, mask21, mask41, mask42, mask421 = self.make_masks(image_seg)
        # sitk.WriteImage(image_seg, '/media/panda/panda/seg.nii.gz')
        # sitk.WriteImage(mask4, '/media/panda/panda/mask4.nii.gz')
        # sitk.WriteImage(mask2, '/media/panda/panda/mask2.nii.gz')
        # sitk.WriteImage(mask1, '/media/panda/panda/mask1.nii.gz')
        # sitk.WriteImage(mask42, '/media/panda/panda/mask42.nii.gz')
        # sitk.WriteImage(mask41, '/media/panda/panda/mask41.nii.gz')
        # sitk.WriteImage(mask421, '/media/panda/panda/mask421.nii.gz')
        # sitk.WriteImage(mask21, '/media/panda/panda/mask21.nii.gz')
        # sitk.WriteImage(image_t1, '/media/panda/panda/t1.nii.gz')
        # sitk.WriteImage(image_t1ce, '/media/panda/panda/t1ce.nii.gz')
        # sitk.WriteImage(image_t2, '/media/panda/panda/t2.nii.gz')
        # sitk.WriteImage(image_flair, '/media/panda/panda/flair.nii.gz')
        ls_temp = []
        ls_1 = self.extract_one_image(items,mask1)
        print(1)
        ls_2 = self.extract_one_image(items,mask2)
        print(2)
        ls_4 = self.extract_one_image(items,mask4)
        print(3)
        ls_21 = self.extract_one_image(items,mask21)
        print(4)
        ls_41 = self.extract_one_image(items,mask41)
        print(5)
        ls_42 = self.extract_one_image(items,mask42)
        print(6)
        ls_421 = self.extract_one_image(items,mask421)
        print(7)
        ls_temp.extend(ls_1)
        ls_temp.extend(ls_2)
        ls_temp.extend(ls_4)
        ls_temp.extend(ls_21)
        ls_temp.extend(ls_41)
        ls_temp.extend(ls_42)
        ls_temp.extend(ls_421)
        for keys in self.ls_survival:
            if keys[0] in items[0]:
                ls_temp.append(keys[1])
                ls_temp.append(keys[2])
#                self.dict_features[keys[0]] = [np.float64(v) for v in ls_temp]
                tmp = [np.float64(v) for v in ls_temp]
                self.save_one_feature(tmp, keys[0])
                self.ls_features.append(tmp)
        
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
        self.dict_features = dict()
        self.dict_all = dict()
        self.ls_features = []
        ls_images = self.read_images()
        # calculate the image features
        n = 0
        for items in ls_images:
            print('s')
            if self.is_feature_calculated(items) == False:
                self.get_one_radiomics(items, title = False)
                #self.save_feature()
            else:
                print('calated')
            n += 1
            print(n)
te = testUI()
te.read_survival_csv()
te.testRadiomics()
te.normalization()
