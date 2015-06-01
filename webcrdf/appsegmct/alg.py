#!/usr/bin/env python
from sys import path_importer_cache

__author__ = 'ar'

import math
import time
import os
import glob
import sys
import numpy as np
import shutil
import multiprocessing as mp
import zipfile
import datetime

import cv2

import nibabel as nib

import matplotlib.pyplot as plt

#################################
# Global definitions
fileNameInput='input'

#################################
def getUniqueDirNameIndex():
    return "userdatact-%s" % datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")

def getFileExt(fname):
    tstr=os.path.splitext(fname)
    s1=tstr[1]
    s2=os.path.splitext(tstr[0])[1]
    sext='%s%s' % (s2,s1)
    return sext

#################################
def task_proc_segmct(data):
    ptrPathWdir     = data[0]
    segmCT = SegmentatorCT(ptrPathWdir)
    segmCT.segmentCTMask()

class TaskManagerSegmCT:
    def __init__(self, nproc=4):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
    def appendTaskSegmCT(self, wdir):
        vdata=[wdir]
        self.pool.apply_async(task_proc_segmct, [vdata] )

#################################
class SegmentatorCT:
    def __init__(self,parWDir):
        self.wdir=parWDir
        self.dirID=os.path.basename(self.wdir)
        self.run_esegm="Eduard_MIPL_Segmentation_CRDF_Nifti_Release_x64 %s %s >%s 2>&1"
        self.fnInput=None
        self.fnSegmented="segmented.nii.gz"
        self.fnPreviewInp="preview.png"
        self.fnPreviewSgm="segmented.png"
        self.fnOutputZip="result.zip"
        self.fnError="err.txt"
        self.fnLOG="log.txt"
        self.findInput()
    """
    find input CT-image file by mask 'input.*' in wdir
    """
    def findInput(self):
        flst=glob.glob("%s/input.*" % self.wdir)
        if len(flst)<1:
            self.fnInput=None
            return False
        self.fnInput=flst[0]
        if not os.path.isfile(self.fnInput):
            self.fnInput=None
            return False
        pathError="%s/%s" % (self.wdir, self.fnError)
        if os.path.isfile(pathError):
            os.remove(pathError)
        return True
    def setWDir(self, parWDir):
        self.wdir=parWDir
        self.dirID=os.path.basename(self.wdir)
        return self.findInput()
    def removeWDir(self):
        if os.path.isdir(self.wdir):
            shutil.rmtree(self.wdir)
    """
    print error to predefined file (err.txt)
    """
    def printError(self, strError, isToStdout=True):
        if os.path.isdir(self.wdir):
            tfErr="%s/%s" % (self.wdir, self.fnError)
            try:
                f=open(tfErr, "a")
                str="Error [%s] : %s\n" % (tfErr, strError)
                f.write(str)
                f.close()
                if isToStdout:
                    print str
            except:
                print "Error: can't create file [%s]" % tfErr
                return False
        else:
            return False
    """
    prepare preview for input CT-image
    """
    def makePreviewInp(self):
        if os.path.isdir(self.wdir):
            if os.path.isfile(self.fnInput):
                try:
                    ndata=nib.load(self.fnInput)
                    if (min(ndata.shape)>40) and (len(ndata.shape)==3):
                        ndata=ndata.get_data()
                        timg=np.rot90(ndata[:,:,ndata.shape[2]/2]).astype(np.float)
                        vMin=-1000.
                        vMax=+200.
                        timg=255.*(timg-vMin)/(vMax-vMin)
                        timg[timg<0]=0
                        timg[timg>255.]=255.
                        timg=cv2.normalize(timg,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
                        timg=cv2.resize(timg,(256,256))
                        fout="%s/%s" % (self.wdir, self.fnPreviewInp)
                        cv2.imwrite(fout, timg)
                        return True
                    else:
                        self.printError("Invalid CT-Image [%s] (bad resolution)" % self.fnInput)
                        return False
                except:
                    self.printError("Can't load CT-Image [%s]" % self.fnInput)
                    return False
            else:
                print "Can't find input file [%s]" % self.fnInput
                return False
        else:
            print "Can't find workdir [%s]" % self.wdir
            return False
    def getUniqueDirNameIndex(self):
        return "userdatact-%s" % datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
    """
    Adjust image brightness by percentile
    """
    def adjustImage(self, img, perc):
        im = img.astype(np.float)
        tbrd=math.floor(0.1*np.min(im.shape))
        imc=im[tbrd:-tbrd, tbrd:-tbrd]
        q0, q1 = np.percentile(imc[:], [perc, 100.0-perc])
        imm=np.max(im[:])
        im=255.*(im-q0)/( (2.0*perc*imm/100.) + q1-q0)
        im[im<0]=0
        im[im>255]=255
        return im
    """
    Main function: segment CT-image and save lung mask
    """
    def segmentCTMask(self):
        pathInp=self.fnInput
        pathOut="%s/%s" % (self.wdir, self.fnSegmented)
        pathZip="%s/%s" % (self.wdir, self.fnOutputZip)
        pathPreviewInp="%s/%s" % (self.wdir, self.fnPreviewInp)
        pathPreviewSgm="%s/%s" % (self.wdir, self.fnPreviewSgm)
        pathLog="%s/%s" % (self.wdir, self.fnLOG)
        if os.path.isdir(self.wdir):
            tret=False
            if not os.path.isfile(pathPreviewInp):
                tret=self.makePreviewInp()
            else:
                tret=True
            if tret:
                strRun=self.run_esegm % (pathInp, pathOut, pathLog)
                os.system(strRun)
                if os.path.isfile(pathOut):
                    try:
                        data1=nib.load(pathOut).get_data()
                        timg2=cv2.imread(pathPreviewInp,1)
                        idx=data1.shape[2]/2
                        timg1=np.rot90(data1[:,:,idx]).astype(np.float)
                        timg1=cv2.resize(timg1, (255,255))
                        tmp=timg2[:,:,2]
                        tmp[timg1>-3000]=255
                        timg2[:,:,2]=tmp
                        cv2.imwrite(pathPreviewSgm,timg2)
                        zObj=zipfile.ZipFile(pathZip, 'w')
                        zipDir='%s_dir' % os.path.basename(self.wdir)
                        lstFimg=(pathInp, pathOut, pathPreviewInp, pathPreviewSgm)
                        for ff in lstFimg:
                            ffbn = os.path.basename(ff)
                            zObj.write(ff, "%s/%s" % (zipDir, ffbn))
                    except:
                        self.printError("Error : can't postprocess CT-Image [%s]" % pathOut)
                else:
                    self.printError("Error in segmentation process [%s]" % strRun)
        else:
            self.printError("Can't find wdir [%s]" % self.wdir)
        ret=os.path.isfile(pathPreviewSgm)
        return ret
    def printInfo(self):
        print "-- Info --"
        print "Wdir:    ", self.wdir
        print "*RUN[segm-CT]:      ", self.run_esegm
    def getLines(self, fname):
        ret=None
        with open(fname,'r') as f:
            ret=f.read().splitlines()
        return ret

#################################
def test_main1():
    def_wdir="/home/ar/data/data_CT_Segm_test/userdata-0"
    segmCT=SegmentatorCT(def_wdir)
    ret=segmCT.makePreviewInp()
    print ret
    ret=segmCT.segmentCTMask()
    print ret

def test_main2():
    lstWDir=['/home/ar/data/data_CT_Segm_test/userdata-1',
             '/home/ar/data/data_CT_Segm_test/userdata-2',
             '/home/ar/data/data_CT_Segm_test/userdata-3',
             '/home/ar/data/data_CT_Segm_test/userdata-4',
             '/home/ar/data/data_CT_Segm_test/userdata-5']
    tmSegmCT=TaskManagerSegmCT(nproc=3)
    for ii in lstWDir:
        tmSegmCT.appendTaskSegmCT(ii)
    tmSegmCT.pool.close()
    tmSegmCT.pool.join()

#################################
if __name__=="__main__":
    test_main1()
    # test_main2()

    # task_proc_segmct((def_wdir, lst_fimg[0]))

