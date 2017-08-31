# -*- coding: utf-8 -*-,
"""
Created on Tue May  9 13:53:20 2017

brain tumor overall survival prediction


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

m_patient_info = dict()
m_image_path = dict()
m_seg_path = dict()
m_radiomics_feature = dict()


# step 1. read the csv survival information
def read_survival_csv(csv_filePath):
    '''read the survival information and age of the patient
    # read the csv file for the surval data'''
    ls_survival = dict()
    # read the csv file
    with open(csv_filePath) as cf:
        spreader = csv.reader(cf, delimiter=',', quotechar='\'')
        for row in spreader:
            if not 'Age' in row: # remove the csv file title
                ls_survival[row[0]] = [row[1], row[2]]
    return ls_survival
csv_filePath = '../1.data/Brats17TrainingData/survival_data.csv'
m_patient_info = read_survival_csv(csv_filePath)
# step 2. read the image data
def read_image_names(data_filePath):
    '''read all the image data in the csv file (not all the images under the folder)'''
    image_dict = dict()
    seg_dict = dict()
    for k, v in m_patient_info.items():
        k_path = data_filePath + k
        ls_files = []
        ls_files = os.listdir(k_path)
        for one_file in ls_files:
            index_end = one_file.rfind('_') - len(one_file)
            index_start = -7
            # key is t1, t2, t1ce, seg, flair
            one_type = one_file[index_end:index_start]
            # if the type is 'seg', store it in the other dict
            if 'seg' in one_type:
                seg_dict[k] = k_path + '/' + one_file
            else:
                k_name = k + one_type
                image_dict[k_name] = k_path + '/' + one_file
    return image_dict, seg_dict

data_filePath = '../1.data/Brats17TrainingData/DATA/'
m_image_path, m_seg_path = read_image_names(data_filePath)


# step 3. image preprocessing for the orientation and the gray level
def normalize_image_origins(image_path, seg_path):
    '''
    normalize the origions to the first images [0, -239, 0]
    '''
    origins = sitk.ReadImage(list(image_path.values())[0]).GetOrigin()
    for v_img in image_path.values():
        images = sitk.ReadImage(v_img)
        if operator.eq(origins, images.GetOrigin()) == False:
            images.SetOrigin(origins)
            sitk.WriteImage(images, v_img)
     for v_seg in seg_path.values():
            segs = sitk.ReadImage(v_seg)
        if operator.eq(origins, segs.GetOrigin()) == False:
            segs.SetOrigin(origins)
            sitk.WriteImage(segs, v_seg)
  
#normalize_image_origins(m_image_path)


# step 4. calculate the radiomics feature
def radiomics_features(image_path, seg_path):
    '''
    extract the radiomics features including first-order, texture, shape
    '''
    params = os.getcwd() + '/' + 'Params.yaml'
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
