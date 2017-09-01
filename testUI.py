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
import sun_radiomics as srad



class testUI():
    '''
    tst
    '''
    def __init__(self):
        '''
        Args:
            image: Input image
            tumour_mask: Binary image containing the GTV
        Returns: dict containing the features for the image
        '''
        self.ls_images = []
        self.strPath = '/media/panda/panda/1.radiomics/3.data/'  #(self.image_path.text())
        lsFiles = os.listdir(self.strPath)
        # load the images
        for files in lsFiles:
           # print(files)
            self.tmpFiles = self.strPath + files            
    def testRadiomics(self):
        ls_images = []
        lsFiles = os.listdir(self.strPath)
        # load the images
        for files in lsFiles[0:5]:
            print(files)
            tmpFiles = self.strPath + files
            images = os.listdir(tmpFiles)
            ls_tmp = [0, 0, 0, 0, 0]
            tmp = []
            for img in images:
                '''the order : t1, t1ce, t2, flair, seg'''
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
        # calculate the image features
        for items in ls_images:
            image_t1 = sitk.ReadImage(items[0])
            image_seg = sitk.ReadImage(items[4])
            npimg = sitk.GetArrayFromImage(image_t1)
            npmask = sitk.GetArrayFromImage(image_seg)

            spacing = image_t1.GetSpacing()

            fe = calculate_all_features(npimg, npmask)
            # features = {}
            # features.update(srad.group1_features(npimg[npmask > 0]))
            # features.update(srad.gray_level_cooccurrence_features(npimg, npmask))




            # img2 = np.array(image)
            # img2 -= np.amin(image)
            # img2, mask2 = srad.clip_to_bounding_box(img2, tumour_mask)
            # img2 *= mask2 > 0
            # features = {}
            # features.update(srad.group1_features(img2[mask2 > 0]))
            # features.update(srad.tumour_features(mask2, [3, 1, 1]))
            # features.update(srad.gray_level_runlength_features(img2, mask2))
            # features.update(srad.gray_level_cooccurrence_features(img2, mask2))
            # features.update(srad.wavelet_features(img2, mask2))
            # print(features)
            print('gg')

        pass
te = testUI()

te.testRadiomics()


