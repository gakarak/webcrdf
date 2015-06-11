#!/usr/bin/ptyhon
__author__ = 'ar'

import numpy as np
import os
import sys

import array

import sklearn.metrics as MT

def readBinData(fname, ptype=np.int64):
    f=open(fname,'r')
    siz=array.array('i')
    siz.read(f,2)
    siz=(siz[1],siz[0])
    data=np.fromfile(f,dtype=ptype)
    data=data.reshape(siz)
    f.close()
    return data

if __name__=='__main__':
    fdsc='/home/ar/img.data/data_histology/004-data-dsc-tot-p2q(16)[1,3,5]-rc.dat'
    fidx='/home/ar/img.data/data_histology/004-data-idx-tot-p2q(16)[1,3,5]-rc.dat'
    idx=readBinData(fidx,ptype=np.float64).astype(np.int)
    dsc=readBinData(fdsc,ptype=np.float64).astype(np.float32)
    dsc1=dsc[1000,:]
    dst=MT.pairwise_distances(dsc, dsc1, metric='l1')
    print idx
    print '----------------'
    print dsc
    print '----------------'
    print dst