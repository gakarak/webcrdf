#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio

from alg import SegmentatorXRayDNN

lstFnImg=['./data/010.png',
          './data/050.png']

pathCaffeSegNet='/home/ar/deep-learning/caffe-segnet.git-build'
pathModelWeights='../data/scripts_segmxrdnn/segnet_xray_weights.caffemodel'
pathOut='./data-out/'

if __name__ == '__main__':
    numImg=len(lstFnImg)
    plt.figure()
    for ii,fimg in enumerate(lstFnImg):
        segmXRdnn = SegmentatorXRayDNN(parCaffeRoot=pathCaffeSegNet)
        tret = segmXRdnn.runSergmentation(fimg, pathOut, pathModelWeights)
        plt.subplot(numImg, 3, ii * 3 + 1)
        plt.imshow(skio.imread(segmXRdnn.fimgProc))
        plt.subplot(numImg, 3, ii * 3 + 2)
        plt.imshow(segmXRdnn.mskOnImg)
        plt.subplot(numImg, 3, ii * 3 + 3)
        plt.imshow(segmXRdnn.imgMasked)
        print ('[%d/%d]: ' % (ii, numImg))
    plt.show()