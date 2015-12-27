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
import skimage as sk
import skimage.transform
import skimage.color
import skimage.draw
import skimage.io

import cv2

import nibabel as nib

import matplotlib.pyplot as plt

#################################
# Global definitions
fileNameInput='input'
DEF_NII_PREVIEW_SIZE_CT=(192,192,60)
DEF_NII_PREVIEW_SIZE_CR=1024

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
def task_proc_ctslice(data):
    ptrPathWdir     = data[0]
    segmCT = CTSlicer(ptrPathWdir)
    segmCT.makePreviewInp(isPreviewStart=False)

class TaskManagerCTSlice:
    def __init__(self, nproc=4):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
    def appendTaskCTSlice(self, wdir):
        vdata=[wdir]
        self.pool.apply_async(task_proc_ctslice, [vdata] )

#################################
class CTSlicer:
    def __init__(self,parWDir):
        self.wdir=parWDir
        self.dirID=os.path.basename(self.wdir)
        self.run_esegm="Eduard_MIPL_Segmentation_CRDF_Nifti_Release_x64 %s %s >%s 2>&1"
        self.fnInput=None
        self.fnSegmented="previewext.nii.gz"
        self.fnPreviewInp="preview.png"
        self.fnPreviewExt="previewext.png"
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
    prepare preview OLD for input CT-image
    """
    def makePreviewInpOld(self):
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
    """
    prepare preview for input CT-image
    """
    def makePreviewInp(self, isPreviewStart=True, sizPreview=(512,512)):
        if os.path.isdir(self.wdir):
            if os.path.isfile(self.fnInput):
                try:
                    if isPreviewStart:
                        fout="%s/%s" % (self.wdir, self.fnPreviewInp)
                        timg=self.makePreviewImageForCTStart(self.fnInput)
                    else:
                        fout="%s/%s" % (self.wdir, self.fnPreviewExt)
                        timg=self.makePreviewImageForCT(self.fnInput)
                        time.sleep(5)
                    timgR=self.resizeImageToSize(timg, tuple(sizPreview))
                    cv2.imwrite(fout, timgR)
                    return True
                    # ndata=nib.load(self.fnInput)
                    # if (min(ndata.shape)>40) and (len(ndata.shape)==3):
                    #     ndata=ndata.get_data()
                    #     timg=np.rot90(ndata[:,:,ndata.shape[2]/2]).astype(np.float)
                    #     vMin=-1000.
                    #     vMax=+200.
                    #     timg=255.*(timg-vMin)/(vMax-vMin)
                    #     timg[timg<0]=0
                    #     timg[timg>255.]=255.
                    #     timg=cv2.normalize(timg,None,0,255,cv2.NORM_MINMAX, cv2.CV_8U)
                    #     timg=cv2.resize(timg,(256,256))
                    #     fout="%s/%s" % (self.wdir, self.fnPreviewInp)
                    #     cv2.imwrite(fout, timg)
                    #     return True
                    # else:
                    #     self.printError("Invalid CT-Image [%s] (bad resolution)" % self.fnInput)
                    #     return False
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
    # """
    # Main function: segment CT-image and save lung mask
    # """
    # def segmentCTMask(self):
    #     pathInp=self.fnInput
    #     pathOut="%s/%s" % (self.wdir, self.fnSegmented)
    #     pathZip="%s/%s" % (self.wdir, self.fnOutputZip)
    #     pathPreviewInp="%s/%s" % (self.wdir, self.fnPreviewInp)
    #     pathPreviewSgm="%s/%s" % (self.wdir, self.fnPreviewExt)
    #     pathLog="%s/%s" % (self.wdir, self.fnLOG)
    #     if os.path.isdir(self.wdir):
    #         tret=False
    #         if not os.path.isfile(pathPreviewInp):
    #             tret=self.makePreviewInp()
    #         else:
    #             tret=True
    #         if tret:
    #             strRun=self.run_esegm % (pathInp, pathOut, pathLog)
    #             os.system(strRun)
    #             if os.path.isfile(pathOut):
    #                 try:
    #                     data1=nib.load(pathOut).get_data()
    #                     timg2=cv2.imread(pathPreviewInp,1)
    #                     idx=data1.shape[2]/2
    #                     timg1=np.rot90(data1[:,:,idx]).astype(np.float)
    #                     timg1=cv2.resize(timg1, (255,255))
    #                     tmp=timg2[:,:,2]
    #                     tmp[timg1>-3000]=255
    #                     timg2[:,:,2]=tmp
    #                     cv2.imwrite(pathPreviewSgm,timg2)
    #                     zObj=zipfile.ZipFile(pathZip, 'w')
    #                     zipDir='%s_dir' % os.path.basename(self.wdir)
    #                     lstFimg=(pathInp, pathOut, pathPreviewInp, pathPreviewSgm)
    #                     for ff in lstFimg:
    #                         ffbn = os.path.basename(ff)
    #                         zObj.write(ff, "%s/%s" % (zipDir, ffbn))
    #                 except:
    #                     self.printError("Error : can't postprocess CT-Image [%s]" % pathOut)
    #             else:
    #                 self.printError("Error in segmentation process [%s]" % strRun)
    #     else:
    #         self.printError("Can't find wdir [%s]" % self.wdir)
    #     ret=os.path.isfile(pathPreviewSgm)
    #     return ret
    def printInfo(self):
        print "-- Info --"
        print "Wdir:    ", self.wdir
        print "*RUN[segm-CT]:      ", self.run_esegm
    def getLines(self, fname):
        ret=None
        with open(fname,'r') as f:
            ret=f.read().splitlines()
        return ret
    def resizeImageToSize(self, img, sizNew, parBorder=0, parInterpolation=2L): # parInterpolation=cv2.INTER_CUBIC
        if len(img.shape)<3:
            img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        sizImg=img.shape
        if (sizNew[0]<2) or (sizNew[1]<2) or (sizImg[0]<2) or (sizImg[1]<2):
            return None
        sizImgf=np.array(sizImg, np.float)
        sizNewf=np.array(sizNew, np.float)
        k1=sizNewf[0]/sizNewf[1]
        k2=sizImgf[0]/sizImgf[1]
        eps=np.abs(k1-k2)/np.abs(k1+k2)
        if eps<0.002:
            if parBorder<1:
                return cv2.resize(img, sizNew, interpolation=parInterpolation)
            else:
                ret=cv2.resize(img, sizNew, interpolation=parInterpolation)
                ret=cv2.copyMakeBorder(ret, parBorder,parBorder,parBorder,parBorder, borderType=cv2.BORDER_CONSTANT, value=0)
                return cv2.resize(ret, sizNew, interpolation=cv2.INTER_CUBIC)
        #
        parScl=sizNewf[0]/sizImgf[0]
        sizImgNewf=np.array( (sizNewf[0], sizImgf[1]*sizNewf[0]/sizImgf[0]),  np.float)
        if (k2<k1):
            sizImgNewf=np.array( (sizImgf[0]*sizNewf[1]/sizImgf[1], sizNewf[1]),  np.float)
            parScl=sizNewf[1]/sizImgf[1]
        dx=(sizNewf[1]-sizImgNewf[1])/2.
        dy=(sizNewf[0]-sizImgNewf[0])/2.
        warpMat=np.zeros((2,3), np.float)
        warpMat[0,0]=parScl
        warpMat[1,1]=parScl
        warpMat[0,2]=+dx
        warpMat[1,2]=+dy
        if parBorder<1:
            return cv2.warpAffine(img, warpMat, sizNew[::-1])
        else:
            ret=cv2.warpAffine(img, warpMat, sizNew[::-1])
            ret=cv2.copyMakeBorder(ret, parBorder,parBorder,parBorder,parBorder, borderType=cv2.BORDER_CONSTANT, value=0)
            return cv2.resize(ret, sizNew[::-1], interpolation=cv2.INTER_CUBIC)
    """
    Normalize CT image by Lung-preset
    """
    def calcNormImageCT(self,img):
        timg=img.astype(np.float)
        vMin=-1000.
        vMax=+200.
        ret=255.*(timg-vMin)/(vMax-vMin)
        ret[ret<0]=0
        ret[ret>255]=255.
        return ret.astype(np.uint8)
    """
    Prepare start-preview for CT-Image on original size
    """
    def makePreviewImageForCTStart(self, fnii, isDebug=False):
        dataHdr=nib.load(fnii)
        data=dataHdr.get_data()
        numFn=data.shape[2]
        sizPad=8
        lstZp=np.linspace(0.9,0.2,4)
        lstImg=[]
        for pp in lstZp:
            tidx=round(pp*numFn)
            lstImg.append(np.pad(self.calcNormImageCT(np.rot90(data[:,:,tidx])), sizPad, 'constant', constant_values=(0)))
        lstImgRGB=[]
        for ii in lstImg:
            lstImgRGB.append(sk.color.gray2rgb(ii))
        imgPH0=np.concatenate((lstImgRGB[0],lstImgRGB[1]), axis=1)
        imgPH1=np.concatenate((lstImgRGB[2],lstImgRGB[3]), axis=1)
        imgPano=np.concatenate((imgPH0,imgPH1))
        if isDebug:
            plt.imshow(imgPano)
            plt.show()
        ret=cv2.cvtColor(imgPano,cv2.COLOR_RGB2BGR)
        return ret
    """
    Prepare preview for CT-Image on original size
    """
    def makePreviewImageForCT(self, fnii, isDebug=False):
        dataHdr=nib.load(fnii)
        data=dataHdr.get_data()
        numFn=data.shape[2]
        sizPad=8
        lstZp=np.linspace(0.8,0.3,3)
        lstImg=[]
        for pp in lstZp:
            tidx=round(pp*numFn)
            lstImg.append(np.pad(self.calcNormImageCT(np.rot90(data[:,:,tidx])), sizPad, 'constant', constant_values=(0)))
        imgX=np.rot90(data[:,data.shape[1]/2,:])
        imgX=sk.transform.resize(imgX.copy(), data.shape[:2], order=4)
        lstImg.append(np.pad(self.calcNormImageCT(imgX), sizPad, 'constant', constant_values=(0)))
        #
        lstImgRGB=[]
        for ii in lstImg:
            lstImgRGB.append(sk.color.gray2rgb(ii))
        lstColors=((0,255,0),(255,255,0),(255,0,0))
        for ii in xrange(len(lstZp)):
            tsiz=imgX.shape
            tdw=42
            tr=int(tsiz[0] - round(tsiz[0]*lstZp[ii]))
            zzRange=range(-3,4,1)
            if imgX.shape[0]>400:
                zzRange=range(-4,5,1)
            for zz in zzRange:
                trr,tcc=sk.draw.line(tr+zz,tdw,tr+zz,tsiz[1]-tdw)
                sk.draw.set_color(lstImgRGB[3],(trr,tcc), lstColors[ii])
        for ii in range(len(lstImgRGB)-1):
            tsiz=lstImgRGB[ii].shape
            trad=12
            if tsiz[0]>400:
                trad=16
            trr,tcc=sk.draw.circle(64,tsiz[1]-64,trad)
            sk.draw.set_color(lstImgRGB[ii],(trr,tcc), lstColors[ii])
        imgPH0=np.concatenate((lstImgRGB[0],lstImgRGB[1]), axis=1)
        imgPH1=np.concatenate((lstImgRGB[2],lstImgRGB[3]), axis=1)
        imgPano=np.concatenate((imgPH0,imgPH1))
        #
        if isDebug:
            plt.imshow(imgPano)
            plt.show()
        ret=cv2.cvtColor(imgPano,cv2.COLOR_RGB2BGR)
        return ret
    def makePreviewNifti(self, fniiInp, fniiOut, newSize=DEF_NII_PREVIEW_SIZE_CT):
        try:
            imgNifti=nib.load(fniiInp)
            data=imgNifti.get_data()
            if data.shape[2]>1 : # 3D-Image
                dataNew=sk.transform.resize(data,newSize,order=4, preserve_range=True)
                oldSize=data.shape
                affineOld=imgNifti.affine.copy()
                affineNew=imgNifti.affine.copy()
                k20_Old=float(oldSize[2])/float(oldSize[0])
                k20_New=float(newSize[2])/float(newSize[0])
                for ii in xrange(3):
                    tCoeff=float(newSize[ii])/float(oldSize[ii])
                    if ii==2:
                        tCoeff=(affineNew[0,0]/affineOld[0,0])*(k20_Old/k20_New)
                    affineNew[ii,ii]*=tCoeff
                    affineNew[ii,3 ]*=tCoeff
                dataNew=self.calcNormImageCT(dataNew)
                niiHdr=imgNifti.header
                niiHdr.set_data_dtype(np.uint8)
                imgNiftiResiz=nib.Nifti1Image(dataNew, affineNew, header=niiHdr)
                nib.save(imgNiftiResiz, fniiOut)
            else: # 2D-Image
                maxSize=np.max(data.shape[:2])
                affineNew=imgNifti.affine.copy()
                if maxSize>DEF_NII_PREVIEW_SIZE_CR:
                    tCoeff=float(DEF_NII_PREVIEW_SIZE_CR)/float(maxSize)
                    maxSize=DEF_NII_PREVIEW_SIZE_CR
                    newSize=(maxSize, (data.shape[1]*maxSize)/data.shape[0], 1)
                    if data.shape[1]>data.shape[0]:
                        newSize=((data.shape[0]*maxSize)/data.shape[1], maxSize, 1)
                    dataNew=sk.transform.resize(data,newSize,order=4, preserve_range=True)
                    affineNew[0,0]*=tCoeff
                    affineNew[1,1]*=tCoeff
                else:
                    dataNew=data
                dataNew=cv2.normalize(dataNew[:,:,0],None,0,255, cv2.NORM_MINMAX,cv2.CV_8U)
                niiHdr=imgNifti.header
                niiHdr.set_data_dtype(np.uint8)
                imgNiftiResiz=nib.Nifti1Image(dataNew, affineNew, header=niiHdr)
                nib.save(imgNiftiResiz, fniiOut)
        except:
            # exitError(RET_NII_RESIZ_AND_SAVE, metaInfo=fniiInp)
            pass

#################################
def test_main1():
    def_wdir="/home/ar/data/data_CT_Segm_test/userdata-0"
    segmCT=CTSlicer(def_wdir)
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
    tmSegmCT=TaskManagerCTSlice(nproc=3)
    for ii in lstWDir:
        tmSegmCT.appendTaskCTSlice(ii)
    tmSegmCT.pool.close()
    tmSegmCT.pool.join()

#################################
if __name__=="__main__":
    # test_main1()
    # test_main2()
    wdir='/home/ar/github.com/webcrdf.git/webcrdf/data/users_ctslice/lajrc3hkwmhahq65juc6uob2kq69ug9d/userdatact-2015_12_27-18_29_22_198871'
    fnii='/home/ar/github.com/webcrdf.git/webcrdf/data/users_ctslice/lajrc3hkwmhahq65juc6uob2kq69ug9d/userdatact-2015_12_27-18_29_22_198871/input.nii.gz'
    sliceCT=CTSlicer(wdir)
    # retImg=sliceCT.makePreviewImageForCTStart(fnii, isDebug=True)
    sliceCT.makePreviewInp(isPreviewStart=False)
    # plt.show()

    # task_proc_segmct((def_wdir, lst_fimg[0]))

