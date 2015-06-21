#!/usr/bin/ptyhon
__author__ = 'ar'

import numpy as np
import json
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

class VideoCBIRSearcher:
    def __init__(self, wdir=None):
        if wdir is not None:
            self.loadDir(wdir)
    def loadDir(self, wdir):
        self.wdir=wdir
        self.lstDSC=[]
        self.fnDataJson="%s/data.json" % wdir
        with open(self.fnDataJson, 'r') as f:
            self.dataJson=json.load(f)
        numVideo=len(self.dataJson)
        for ii in xrange(numVideo):
            tfdsc='%s/%s-dsc.dat' % (self.wdir, self.dataJson[ii][0])
            tdsc=readBinData(tfdsc,np.float32)
            self.lstDSC.append(tdsc)
    def printInfo(self):
        print "---- JSON ----"
        for ii in self.dataJson:
            print ii[:4]
        for ii in self.lstDSC:
            print ii.shape
    def getSelectedDSC(self, idxVideo, idxFrame):
        ret=self.lstDSC[idxVideo][idxFrame,:]
        return ret
    def getListNGBH(self, idxVideo, idxFrame):
        tdsc=self.getSelectedDSC(idxVideo, idxFrame)
        lstDST=[]
        for ii in xrange(len(self.lstDSC)):
            tdst=MT.pairwise_distances(self.lstDSC[ii], tdsc, metric='l1')[:,0]
            tdst=(255*(2.0-tdst)/2.0).astype(np.int)
            tdst[tdst<0]=0
            tdst[tdst>=255]=255
            lstDST.append(tdst.tolist())
        return lstDST
    def processSelection(self, idxVideo, idxFrame):
        return self.getListNGBH(idxVideo,idxFrame)

if __name__=='__main__':
    wdir='/home/ar/github.com/webcrdf.git/webcrdf/data/datadb.videocbir'
    cbirSearch=VideoCBIRSearcher(wdir)
    cbirSearch.printInfo()
    retDST=cbirSearch.processSelection(2,100)
    for ii in retDST:
        print ii[:10]

