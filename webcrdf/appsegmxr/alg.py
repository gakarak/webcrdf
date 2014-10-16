#!/usr/bin/env python
__author__ = 'ar'

import math
import time
import os
import sys
import numpy as np
import cv2
from skimage import io
import skimage.filter
import sklearn.metrics as MT
import shutil
import multiprocessing as mp
import zipfile

import matplotlib.pyplot as plt

# def_wdir="/home/ar/work/data_all_he2_x256"
# def_fimg="/home/ar/work/data_all_he2_x256/008.png"
# def_fmsk="/home/ar/work/data_all_he2_x256/008.bmp"

# lst_fimg=[
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/001.png",
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/002.png",
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/003.png",
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/004.png",
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/006.png",
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/008.png",
#     "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/009.png"
# ]

lst_fimg=[
    "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/001.png",
    "/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/002.png",
]

def_wdir="/home/ar/big.data/data_xr_in_web2/data_all_he2_x256"
# def_fimg="/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/008.png"
# def_fmsk="/home/ar/big.data/data_xr_in_web2/data_all_he2_x256/008.bmp"

def_fimg="/home/ar/big.data/data.CRDF-CT/1/xray_2.jpg"

#################################
def task_proc_segmxr_bk(data):
    ptrRegClass     = data[0]
    ptrPathImg      = data[1]
    retMsk,retCorr  = ptrRegClass.registerMask(ptrPathImg)
    print "retCorr = %s" % retCorr

def task_proc_segmxr(data):
    ptrPathWdir     = data[0]
    ptrPathImg      = data[1]
    regXR           = RegisterXray()
    regXR.loadDB(ptrPathWdir)
    # regXR.printInfo()
    retMsk,retCorr  = regXR.registerMask(ptrPathImg)
    pathImgMask     = "%s_mask.png"   % ptrPathImg
    pathImgMasked   = "%s_masked.png" % ptrPathImg
    if retCorr>regXR.threshCorrSum:
        cv2.imwrite(pathImgMask,   regXR.newMsk)
        cv2.imwrite(pathImgMasked, regXR.newImgMsk)
    else:
        tmpNewImgMsk = cv2.imread(ptrPathImg, 1) #cv2.IMREAD_COLOR)
        p00=(0,0)
        p01=(0,tmpNewImgMsk.shape[0])
        p10=(tmpNewImgMsk.shape[1],0)
        p11=(tmpNewImgMsk.shape[1], tmpNewImgMsk.shape[0])
        cv2.line(tmpNewImgMsk, p00, p11, (0,0,255), 4)
        cv2.line(tmpNewImgMsk, p01, p10, (0,0,255), 4)
        regXR.newMsk[:]=0
        cv2.imwrite(pathImgMask,   regXR.newMsk)
        cv2.imwrite(pathImgMasked, tmpNewImgMsk)
        fnErr="%s.err" % ptrPathImg
        f=open(fnErr,'w')
        f.close()
    fzip="%s.zip" % ptrPathImg
    zObj=zipfile.ZipFile(fzip, 'w')
    zipDir='%s_dir' % os.path.basename(ptrPathImg)
    lstFimg=(ptrPathImg, pathImgMask, pathImgMasked)
    for ff in lstFimg:
        ffbn = os.path.basename(ff)
        zObj.write(ff, "%s/%s" % (zipDir, ffbn))
    print "retCorr = %s" % retCorr

def task_proc_segmxr2(data):
    ptrPathWdir     = data[0]
    ptrPathImg      = data[1]
    regXR           = RegisterXray()
    regXR.loadDB(ptrPathWdir)
    # regXR.printInfo()
    retMsk,retCorr,sumPts,ptsXY  = regXR.registerMask(ptrPathImg)
    pathImgMask     = "%s_mask.png"   % ptrPathImg
    pathImgMasked   = "%s_masked.png" % ptrPathImg
    pathImgOnMask   = "%s_onmask.png" % ptrPathImg
    pathPtsCSV      = "%s_pts.csv"    % ptrPathImg
    if retCorr>regXR.threshCorrSum:
        cv2.imwrite(pathImgMask,   regXR.newMsk)
        cv2.imwrite(pathImgMasked, regXR.newImgMsk)
        cv2.imwrite(pathImgOnMask, regXR.newImgOnMsk)
        np.savetxt(pathPtsCSV, regXR.pts, delimiter=',', newline='\n', fmt='%0.1f')
    else:
        tmpNewImgMsk = cv2.imread(ptrPathImg, 1) #cv2.IMREAD_COLOR)
        tmpImgOnMsk  = np.zeros((tmpNewImgMsk.shape[0],tmpNewImgMsk.shape[1]), np.uint8)
        p00=(0,0)
        p01=(0,tmpNewImgMsk.shape[0])
        p10=(tmpNewImgMsk.shape[1],0)
        p11=(tmpNewImgMsk.shape[1], tmpNewImgMsk.shape[0])
        cv2.line(tmpNewImgMsk, p00, p11, (0,0,255), 4)
        cv2.line(tmpNewImgMsk, p01, p10, (0,0,255), 4)
        regXR.newMsk[:]=0
        cv2.imwrite(pathImgMask,   regXR.newMsk)
        cv2.imwrite(pathImgMasked, tmpNewImgMsk)
        cv2.imwrite(pathImgOnMask, tmpImgOnMsk )
        tmpPts=np.zeros((7,2), np.float64)
        np.savetxt(tmpPts, delimiter=',', newline='\n')
        fnErr="%s.err" % ptrPathImg
        f=open(fnErr,'w')
        f.close()
    fzip="%s.zip" % ptrPathImg
    zObj=zipfile.ZipFile(fzip, 'w')
    zipDir='%s_dir' % os.path.basename(ptrPathImg)
    lstFimg=(ptrPathImg, pathImgMask, pathImgMasked, pathImgOnMask, pathPtsCSV)
    for ff in lstFimg:
        ffbn = os.path.basename(ff)
        zObj.write(ff, "%s/%s" % (zipDir, ffbn))
    print "retCorr = %s" % retCorr


class TaskManagerSegmXR:
    def __init__(self, nproc=4):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
        self.regXR = RegisterXray()
    def loadData(self, wdir):
        self.wdir = wdir
        self.regXR.loadDB(wdir)
        self.regXR.printInfo()
    def appendTaskSegmXR(self, fimg):
        vdata=[self.wdir, fimg]
        # self.pool.apply_async(task_proc_segmxr, [vdata] )
        self.pool.apply_async(task_proc_segmxr2, [vdata] )

#################################
class RegisterXray:
    def __init__(self):
        self.numNGBH=5
        self.siz=256
        self.wsiz=(self.siz, self.siz)
        self.fnDscShort="dsc.csv"
        self.fnIdxShort="idx.csv"
        self.fnParamShort="parameters_BSpline.txt"
        self.run_elastix="elastix -f %s -m %s -p %s -out %s -threads 4 >/dev/null"
        self.run_transformix="transformix -in %s -tp %s/TransformParameters.0.txt -out %s >/dev/null"
        self.fnResImgShort="result.0.bmp"
        self.fnResMskShort="result.bmp"
        self.newMsk=None
        self.newImgMsk=None
        self.pts=None
        self.newImgOnMsk=None
        self.threshCorrSum = 0.7
    def loadDB(self, parDirDB):
        self.wdir = parDirDB
        if not os.path.isdir:
            print "Can't find directory [%s]" % self.wdir
            return False
        self.odir = "%s-out" % self.wdir
        try:
            os.mkdir(self.odir)
        except:
            pass
        if not os.path.isdir(self.odir):
            print "Can't create out-directory [%s]" % self.odir
            return False
        tmpFnDscFull="%s/%s" % (self.wdir, self.fnDscShort)
        tmpFnIdxFull="%s/%s" % (self.wdir, self.fnIdxShort)
        tmpFnParamFull="%s/%s" % (self.wdir, self.fnParamShort)
        if not os.path.isfile(tmpFnDscFull):
            print "Ca't find DSC-file in DB [%s]" % tmpFnDscFull
            return False
        if not os.path.isfile(tmpFnIdxFull):
            print "Can't find Index-file in DB [%s]" % tmpFnIdxFull
            return False
        if not os.path.isfile(tmpFnParamFull):
            print "Can't find Elastix-Param-file in DB [%s]" % tmpFnParamFull
            return False
        self.fnDsc=tmpFnDscFull
        self.fnIdx=tmpFnIdxFull
        self.fnParam=tmpFnParamFull
        self.dataDsc=np.genfromtxt(self.fnDsc, delimiter=',')
        self.dataIdx=self.getLines(self.fnIdx)
        self.dataFnImg=[]
        self.dataFnMsk=[]
        for idx in self.dataIdx:
            tmpfImg='%s/%s.png' % (self.wdir, idx)
            tmpfMsk='%s/%s.bmp' % (self.wdir, idx)
            if not os.path.isfile(tmpfImg):
                print "Error: Can't find Image-file in DB [%s]" % tmpfImg
                return False
            if not os.path.isfile(tmpfMsk):
                print "Error: Can't find Mask-file in DB [%s]" % tmpfMsk
                return False
            self.dataFnImg.append(tmpfImg)
            self.dataFnMsk.append(tmpfMsk)
        if len(self.dataIdx)!=len(self.dataDsc):
            print "Error: Incorrect Index or DSC data"
            return False
        return True
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
    def calcDscRadon(self, img):
        if len(img.shape)!=2:
            print "Error: bad image format: ", img.shape
            return None
        dsc=np.concatenate((np.sum(img,axis=0), np.sum(img,axis=1)))
        dsc=dsc/(img.shape[0]*img.shape[1])
        return dsc
    def getOutDir(self):
        return "%s/out-%d" % (self.odir, time.time()*1000)
    def helperMkDir(self, dirName):
        try:
            os.mkdir(dirName)
        except:
            pass
        if not os.path.isdir(dirName):
            print "Error: Can't create directory [%s]" % dirName
            return False
        else:
            return True
    def findNGBH(self, img):
        dsc=self.calcDscRadon(img)
        tdst=MT.pairwise_distances(self.dataDsc, [dsc], metric='correlation')[:,0]
        sidx=np.argsort(tdst)[:self.numNGBH]
        return sidx
    """
    Main function: return registered mask
    """
    def registerMask(self, fimg, isRemoveDir=False, fmsk=None):
        # img=cv2.imread(fimg, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        img=cv2.imread(fimg, 0).astype(np.float64)
        siz0=img.shape
        img=cv2.resize(img, self.wsiz)
        img=self.adjustImage(img, 1.0)
        retNgbh=self.findNGBH(img)
        tdirOut=self.getOutDir()
        self.helperMkDir(tdirOut)
        toutMsk="%s/%s" % (tdirOut, self.fnResMskShort)
        toutImg="%s/%s" % (tdirOut, self.fnResImgShort)
        tfin="%s/fix.png" % tdirOut
        cv2.imwrite(tfin, np.uint8(img))
        sumMask=None
        sumPts =None
        sumCorr=0.0
        maxCorr=-100
        for ii in retNgbh:
            # print "prcess %d : %s" % (ii, self.dataFnImg[ii])
            tfmovImg=self.dataFnImg[ii]
            tfmovMsk=self.dataFnMsk[ii]
            tfmovPts='%s_pts.png' % tfmovImg
            strRun0=self.run_elastix     % (tfin, tfmovImg, self.fnParam, tdirOut)
            strRun1=self.run_transformix % (tfmovMsk, tdirOut, tdirOut)
            os.system(strRun0)
            os.system(strRun1)
            # tmsk=cv2.imread(toutMsk, cv2.IMREAD_GRAYSCALE).astype(np.float)/255.0
            tmsk=cv2.imread(toutMsk, 0).astype(np.float)/255.0
            tmsk=skimage.filter.gaussian_filter(tmsk, 0.5)
            # timg=cv2.imread(toutImg, cv2.IMREAD_GRAYSCALE).astype(np.float)
            timg=cv2.imread(toutImg, 0).astype(np.float)
            curCorr=np.corrcoef(img[20:-20].reshape(-1), timg[20:-20].reshape(-1))[0,1]
            sumCorr+=curCorr
            if sumMask==None:
                sumMask=tmsk
            else:
                sumMask+=tmsk
            #### PTS
            strRun2=self.run_transformix % (tfmovPts, tdirOut, tdirOut)
            os.system(strRun2)
            tpts=cv2.imread(toutMsk, 0)
            if maxCorr<curCorr:
                sumPts=tpts.copy()
                maxCorr=curCorr
        sumCorr/=self.numNGBH
        ret=(sumMask/self.numNGBH)
        ret=cv2.resize(ret,(siz0[1],siz0[0]), interpolation=2)
        ret=(ret>0.5)
        ret=255*np.uint8(ret)
        ### PTS-array (prepare)
        numPts=7
        ptsXY=np.zeros((numPts,2), np.float64)
        kxy=(ret.shape[1]/256., ret.shape[0]/256.)
        for pp in xrange(0,7):
            pvalue=100+20*pp
            xy=np.where(sumPts==pvalue)
            if len(xy[0])>0:
                xym=np.mean(xy,1)
                ptsXY[pp,0]=xym[1]*kxy[0]
                ptsXY[pp,1]=xym[0]*kxy[1]
            else:
                ptsXY[pp,0]=-1.0
                ptsXY[pp,1]=-1.0
        ### PTS-array
        if isRemoveDir:
            shutil.rmtree(tdirOut)
        self.newMsk=ret
        self.pts=ptsXY.copy()
        self.newImgMsk=self.makeMaskedImage(fimg, self.newMsk)
        self.newImgOnMsk=self.makeImgOnMask(fimg, self.newMsk)
        return (ret,sumCorr,sumPts,ptsXY)
    def makeMaskedImage(self, fimg, msk):
        img=cv2.imread(fimg, 1) #cv2.IMREAD_COLOR)
        tmp=img[:,:,2]
        tmp[msk>0]=255
        img[:,:,2]=tmp
        if self.pts != None:
            numPts=self.pts.shape[0]
            scaleSiz=np.min( (img.shape[0],img.shape[1]) )
            prad=int(5.*scaleSiz/256.)
            for pp in xrange(0,numPts):
                cpp=self.pts[pp]
                cv2.circle(img, (int(cpp[0]), int(cpp[1])), prad+2, (255,255,255), -1) # cv2.cv.FILLED
                cv2.circle(img, (int(cpp[0]), int(cpp[1])), prad+0, (0,0,255), -1) # cv2.cv.FILLED
        return img
    def makeImgOnMask(self, fimg, msk):
        tmpImg=cv2.imread(fimg, 0).astype(np.int32)
        tmpImg=tmpImg+1
        tmpImg[tmpImg>255]=255
        tmpImg[msk==0]=0
        tmpImg=tmpImg.astype(np.uint8)
        return tmpImg
    def printInfo(self):
        print "-- Info --"
        print "Wdir:    ", self.wdir
        print "Dsc:     ", self.fnDsc
        print "Idx:     ", self.fnIdx
        print "Params:  ", self.fnParam
        print "DB-Size: ", len(self.dataIdx)
        print "*RUN[elatix]:      ", self.run_elastix
        print "*RUN[transformix]: ", self.run_transformix
    def getLines(self, fname):
        ret=None
        with open(fname,'r') as f:
            ret=f.read().splitlines()
        return ret

#################################
def test_main0():
    wdir='/home/ar/big.data/dev/work.django/test_pts/datadb.jsrt'
    # wdir='/home/ar/data/data.crdf/test_pts/datadb.jsrt'
    regXRay=RegisterXray()
    regXRay.loadDB(wdir)
    tst1='/home/ar/big.data/dev/work.django/test_pts/ex/003.png'
    # tst1='/home/ar/data/data.crdf/test_pts/ex/003.png'
    ret=regXRay.registerMask(tst1)
    retPts=ret[3]
    fptsOut="%s_pts.csv" % tst1
    resImg=cv2.imread(tst1, 0)
    # for ii in xrange(0,retPts.shape[0]):
    #     cv2.circle(resImg, (int(retPts[ii,0]), int(retPts[ii,1])), 5, (255,255,255))
    np.savetxt(fptsOut, retPts, delimiter=',', newline='\n')
    plt.imshow(resImg)
    plt.show()
    sys.exit(0)

#################################
if __name__=="__main__":
    def_wdir='/home/ar/big.data/dev/work.django/test_pts/datadb.segmxr2'
    lst_fimg=('/home/ar/big.data/dev/work.django/test_pts/ex2/001.png', '/home/ar/big.data/dev/work.django/test_pts/ex2/003.png')
    task_proc_segmxr2((def_wdir, lst_fimg[0]))

    sys.exit(0)
    taskManager = TaskManagerSegmXR(2)
    taskManager.loadData(def_wdir)
    for ff in lst_fimg:
        print "append task [%s]" % ff
        taskManager.appendTaskSegmXR(ff)
        time.sleep(1)
    taskManager.pool.close()
    taskManager.pool.join()
    # xreg=RegisterXray()
    # xreg.loadDB(def_wdir)
    # xreg.printInfo()
    # retMsk,retCorr=xreg.registerMask(def_fimg, isRemoveDir=True)
    # print "corr=", retCorr
    # cv2.imshow("win", xreg.newImgMsk)
    # cv2.waitKey(0)

