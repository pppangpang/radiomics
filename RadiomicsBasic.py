# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:53:20 2017

brain tumor overall survival prediction project


@author: sunpan

"""



import os
import csv
import numpy as np
import nibabel as nib
import numpy.ma as ma
from radiomics import featureextractor
import six
import SimpleITK as sitk
import operator
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline





class RadiomicsFeature():
    
    ''' 
    the basic types used in the radiomcis
    based on one patient information
    '''
    def __init__(self, patientID, patient_info = [], radiomics_info = []):
        self.p_ID = patientID
        self.p_info = patient_info
        self.p_radiomics = radiomics_info

class RadiomicsBasic():
    '''
    this class is the main calculation class for radiomcis features
    '''
    def __init__(self, imagePath, csvPath, calnumber = 3):
        '''initilazitaion the class'''
        self.image_path = imagePath    # the brain tumor image path
        self.csv_path = csvPath     # the patient suvival time
        self.calnum = calnumber

        self.ls_image_keys = ['t1', 't1ce', 't2', 'flair', 'seg'] # the image types in one subject
        self.tags = [0, 1, 2, 4] # the image targes in the seg.nii image file
        
    def read_image_path_API(self):   
        '''the API for read all the image pathes which have survival days'''           
        ls_survival_info = self.read_survival_csv()   
       
    def image_preprocessing_API(self):
        '''
        the API for preprocessing images
        '''        
        self.change_normalize_image_origins()
        
   
    def calculate_radiomics_feature_API(self):
        '''
        the API for caculate radiomics features
        '''
        self.radiomics_features()
        self.add_age_as_radiomics_feature()  
        
    def feature_selection_API(self):
        '''
        the API for feature selction
        '''
        self.select_best_feature()
        
     
    def ls_files(self, filePath):
        ''' find all the brain tumor files (HGG and LGG)'''
        ls_files = []
        ls_files = os.listdir(filePath)
        return ls_files
    
    
    def read_survival_csv(self):
        '''read the survival information and age of the patient'''
        ls_survival = []   
        # read the csv file
        with open(self.csv_path) as cf:
            spreader = csv.reader(cf, delimiter=',', quotechar='\'')
            for row in spreader:
                if not 'Age' in row: # remove the csv file title
                    ls_survival.append(row)
                        
        # read the image path
        for i_info in ls_survival[0:self.caltmp]:
            print('reading the subject : ' + str(i_info[0]))                      
            dict_image_path = dict()
            for i_type in self.ls_image_keys:
                i_type_image = self.read_survival_image_paths(self.image_path + i_info[0], i_type)
                dict_image_path[i_type] = i_type_image
    
            self.patient_info.append(RadiomicsFeature(i_info[0], dict_image_path,i_info[2], i_info[1]))   
       
            return ls_survival
  
    def read_survival_image_paths(self, filepath, image_type):
        '''read the images in a certain type'''
        if os.path.exists(filepath):
            image_path = []
            ls_one_files = self.ls_files(filepath)
            for i in ls_one_files:
                if image_type in i:
                    image_path = filepath + '/' + i
                    return image_path
            
  
    def radiomics_features(self):
        '''
        extract the radiomics features including first-order, texture, shape
        '''
        params = self.working_path + '/' + 'Params.yaml'
        for n_count, i_info in enumerate(self.patient_info[0:self.caltmp]):
            print('extracting radiomics feature for subject ' + str(n_count) + ' : ' + str(i_info.p_ID))
            seg = i_info.p_path[self.ls_image_keys[4]]
            dict_feature = dict()
            ls_feature = []
            for i_key in self.ls_image_keys:
                if not i_key == self.ls_image_keys[4]:
                    img = i_info.p_path[i_key]             
                    extractor = featureextractor.RadiomicsFeaturesExtractor(params)
                    a, b = os.path.split(seg)
                    c, d = os.path.split(img)
                    if operator.eq(a, c) == True:
                        print(b, d)
                    else:
                        print('bad')
                    results = extractor.execute(img, seg)
                    dict_feature[i_key] = results
                    for k, v in six.iteritems(results):
                        ls_feature.append(v)
                        print(k, v)
      
            self.patient_info[n_count].p_radiomics = dict_feature
            self.patient_info[n_count].p_classification = ls_feature
   


    def change_normalize_image_origins(self):
        '''
        normalize the origions to the first images [0, -239, 0]
        '''
        #origins = sitk.ReadImage(self.ls_images_path[0]['t1']).GetOrigin()
        for pre_count, pre_image in enumerate(self.patient_info):
            for pre_keys in self.ls_image_keys:
                #print(pre_image[pre_keys])
                images = sitk.ReadImage(pre_image.p_path[pre_keys])
                #sitk.Show(images)
                #print(pre_count)
                #if operator.eq(origins, images.GetOrigin)  == False:
                    #images.SetOrigin(origins)
                    #sitk.WriteImage(images, pre_image[pre_keys])
                    #print(origins)            
                #print(images.GetOrigin())
                #print(images.GetSpacing())
                #print(images.GetDirection())
                #print(images.GetDimension())
                #print(images.GetNumberOfComponentsPerPixel())

    def add_age_as_radiomics_feature(self):
        '''
        add age as a radiomics feature
        '''
        for n_count, i_info in enumerate(self.patient_info):
            self.patient_info[n_count].p_classification.append(i_info.p_age)

    def unzip_classification_feature(self):
        x = [] # save the classification features
        y = [] # save the survival time
        for i in self.patient_info:
            x.append(i.p_classification)
            y.append(i.p_time)
        return x, y

    def select_best_feature(self):
        x, y = self.unzip_classification_feature()
        anova_filter = SelectKBest(f_regression, k = 3)
        clf = svm.SVC(kernel = 'linear')
        anova_svm = make_pipeline(anova_filter, clf)
        anova_svm.fit(x[0:20], y[0:20])
        anova_svm.predict(x[20:30])























