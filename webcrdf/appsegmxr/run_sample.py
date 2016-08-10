#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import matplotlib.pyplot as plt
import skimage.io as skio
from alg import RegisterXray, task_proc_segmxr2

wdir='../data/datadb.segmxr'
lstFnImg=['data/001_0004FED7_an.dcm.png',
          'data/010_0000BB72_an.dcm.png']

if __name__ == '__main__':
    numImg = len(lstFnImg)
    plt.figure()
    for ii, fimg in enumerate(lstFnImg):
        task_proc_segmxr2([wdir, fimg])
        fimgOrig    = fimg
        fimgOnMask  = '%s_onmaskxr.png' % fimgOrig
        fimgMasked  = '%s_maskedxr.png' % fimgOrig
        plt.subplot(numImg, 3, ii * 3 + 1)
        plt.imshow(skio.imread(fimgOrig), cmap=plt.gray())
        plt.subplot(numImg, 3, ii * 3 + 2)
        plt.imshow(skio.imread(fimgOnMask))
        plt.subplot(numImg, 3, ii * 3 + 3)
        plt.imshow(skio.imread(fimgMasked))
        print ('[%d/%d]: ' % (ii, numImg))
    plt.show()
