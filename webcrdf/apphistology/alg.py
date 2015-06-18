#!/usr/bin/ptyhon
__author__ = 'ar'

import numpy as np
import os
import sys
import json
import time
import cv2
import shutil

import matplotlib.pyplot as plt
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

class HistologySearcher:
    def __init__(self, wdir=None):
        if wdir is not None:
            self.loadDir(wdir)
    def loadDir(self, wdir):
        self.wdir=wdir
        self.fnDataJson="%s/data.json" % wdir
        self.fnDataDSC="%s/dsc.dat" % wdir
        self.fnDataIDX="%s/idx.dat" % wdir
        self.dataIDX=readBinData(self.fnDataIDX, ptype=np.float64).astype(np.int)
        self.dataDSC=readBinData(self.fnDataDSC, ptype=np.float32)
        # self.idxShifts=np.zeros(self.dataIDX.shape[0],np.int)
        with open(self.fnDataJson, 'r') as f:
            self.dataJson=json.load(f)
        numSlides=len(self.dataJson)
        self.idxShifts=np.zeros(numSlides,np.int)
        tshift=0
        for ii in xrange(numSlides):
            dshift=self.dataJson[ii][1][0]*self.dataJson[ii][1][1]
            # self.idxShifts[tshift:tshift+dshift]=tshift
            self.idxShifts[ii]=tshift
            tshift+=dshift
    def printInfo(self):
        print "---- JSON ----"
        for ii in self.dataJson:
            print ii
        print "---- DSC ----"
        print "dsc-size: [%dx%d]" % self.dataDSC.shape
        print self.dataDSC
        print "---- IDX ----"
        print "idx-size: [%dx%d]" % self.dataIDX.shape
        print self.dataIDX
        print "---- IDX-SHIFT ----"
        print self.idxShifts
    """
    numeration idxSlide, posr, posc like in MATLAB: started from 1
    """
    def getLinearIdx(self, idxSlide, posr, posc):
        # tidx=self.dataIDX[np.where(self.dataIDX[:,0]==idxSlide),:][0]
        pidx=self.dataJson[idxSlide-1][1][1]*(posr-1)+posc-1
        linIdx=self.idxShifts[idxSlide-1] + pidx
        # print self.dataIDX[linIdx,:]
        # print tidx[pidx,:]
        return linIdx
    def getSelectedDSC(self, idxSlide, posr, posc):
        tidx=self.getLinearIdx(idxSlide,posr,posc)
        ret=self.dataDSC[tidx,:]
        # print ret
        return ret
    def getImgFileName(self, idxSlide, posr, posc):
        return "%s/%03d_%03d.jpg" % (self.dataJson[idxSlide-1][3], posr, posc)
    def getListNGBH(self, idxSlide, posr, posc, numNGBH=8):
        linIdx=self.getLinearIdx(idxSlide, posr, posc)
        tdsc=self.dataDSC[linIdx,:]
        tdst=MT.pairwise_distances(self.dataDSC, tdsc, metric='l1')[:,0]
        tdstSortIdx=np.argsort(tdst)[1:numNGBH+1]
        ret=[]
        for ii in xrange(numNGBH):
            tidxS=tdstSortIdx[ii]
            tdstS=tdst[tidxS]
            tpos=self.dataIDX[tidxS,:]
            tfn='%s/%03d_%03d.jpg' % (self.dataJson[tpos[0]-1][3], tpos[1], tpos[2])
            # tfn="%s/%s" % (self.wdir, tfn)
            ret.append((tfn, tdstS))
            # print tpos
        # print tdstSortIdx
        # print ret
        return ret
    def genTimedID(self, idx):
        return "%d" % (time.time()*1000)
    def generateDstSegmentation(self, idxSlide, posr, posc, odir):
        if os.path.isdir(odir):
            shutil.rmtree(odir)
        os.mkdir(odir)
        numSlides=len(self.dataJson)
        selDSC=self.getSelectedDSC(idxSlide, posr, posc)
        lstURL=[]
        for ii in xrange(numSlides):
            tsiz=self.dataJson[ii][1]
            tDSC=self.dataDSC[np.where(self.dataIDX[:,0]==(ii+1)),:][0]
            tdst=MT.pairwise_distances(tDSC, selDSC, metric='l1')[:,0]
            tdst=tdst.reshape(tsiz)
            tdstMin=0.0 #np.min(tdst)
            tdstMax=2.0 #np.max(tdst)
            tdst=(tdst-tdstMin)/(tdstMax-tdstMin)
            cmap = plt.get_cmap('bwr')
            tdstRGB=(255*cmap(tdst)).astype(np.uint8)
            #
            tTimeID=self.genTimedID(ii)
            tURL="%s/%s-%s.jpg" % (os.path.basename(odir), self.dataJson[ii][0], tTimeID)
            lstURL.append(tURL)
            foutImg="%s/%s-%s.jpg" % (odir, self.dataJson[ii][0], tTimeID)
            cv2.imwrite(foutImg, tdstRGB)
        return lstURL
    def processSelection(self, idxSlide, posr, posc, odir):
        retNGB=self.getListNGBH(idxSlide, posr, posc)
        retDST=self.generateDstSegmentation(idxSlide, posr, posc, odir)
        return (retNGB, retDST)

if __name__=='__main__':
    # wdir='/media/Elements/data_histology/datadb.histology'
    wdir='/home/ar/github.com/webcrdf.git/webcrdf/data/datadb.histology'
    histoSearch=HistologySearcher(wdir)
    # histoSearch.printInfo()
    # histoSearch.getLinearIdx(3,30,150)
    # histoSearch.getSelectedDSC(5,30,150)
    # print "%s/%s" % (histoSearch.wdir, histoSearch.getImgFileName(5,30,50))
    # print histoSearch.getListNGBH(5,30,50)
    print histoSearch.processSelection(5,30,50, '/home/ar/tmp/qqq')
    # print histoSearch.dataJson[0]

    # fdsc='/home/ar/img.data/data_histology/004-data-dsc-tot-p2q(16)[1,3,5]-rc.dat'
    # fidx='/home/ar/img.data/data_histology/004-data-idx-tot-p2q(16)[1,3,5]-rc.dat'
    # idx=readBinData(fidx,ptype=np.float64).astype(np.int)
    # dsc=readBinData(fdsc,ptype=np.float64).astype(np.float32)
    # dsc1=dsc[1000,:]
    # dst=MT.pairwise_distances(dsc, dsc1, metric='l1')
    # print idx
    # print '----------------'
    # print dsc
    # print '----------------'
    # print dst