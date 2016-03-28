#!/usr/bin/env python
__author__ = 'ar'

import os
import glob
import multiprocessing as mp
import datetime
import nibabel as nib
import numpy as np
import cv2
import dicom as dcm

import appsegmxr.alg as algxr
import appsegmct.alg as algct


fileNameInputCT='inputct'
fileNameInputXR_Orig='inputxrorig'
fileNameInputXR_uint8='inputxr_uint8.png'
fileNameInputCT_uint8='inputct_uint8.png'

#################################
class ResultReader:
    def __init__(self):
        self.nameRES  = 'res.txt'
        self.nameERR  = 'err.txt'
        self.namePRG  = 'progress.txt'
    def checkWDir(self, wdir):
        ret=getDataNamesCTXR(wdir)
        if (ret[0]!=None) and (ret[1]!=None):
            return True
        else:
            return False
    """
    return: (isFinished, ErrCode, ErrStr)
        Errcode '0' -> all is Ok
    """
    def readResultQuick(self, wdir):
        if not os.path.isdir(wdir):
            return (False,1,'Error: input dir does not exist', 0)
        tfErr="%s/%s" % (wdir,self.nameERR)
        if os.path.exists(tfErr):
            f=open(tfErr,'r')
            txtErr = f.readline()
            f.close()
            return (True,1,txtErr,0)
        # wdir=os.path.dirname(pathImg)
        fnRES='%s/%s' % (wdir, self.nameRES)
        fnPRG='%s/%s' % (wdir, self.namePRG)
        progress=getProgress(fnPRG)
        if os.path.isfile(fnRES):
            return (True,  0, '', progress)
        else:
            return (False, 0, '', progress)
    def readResult(self, wdir):
        tmpRet=self.readResultQuick(wdir)
        fnTEXT='%s/%s' % (wdir, self.nameRES)
        data=[]
        dataQuick=None
        dataQuickP=None
        if tmpRet[0] and (tmpRet[1]==0):
            f=open(fnTEXT)
            dataQuick=f.readline()
            dataQuickP=f.readline()
            data=f.readlines()
            f.close()
        if data!=None:
            data=''.join(data)
        imgIdx=os.path.basename(wdir)
        fnProgress="%s/%s" % (wdir, self.namePRG)
        progress=getProgress(fnProgress)
        return {'isFinished': tmpRet[0], 'errCode':tmpRet[1], 'errStr':tmpRet[2], 'idx':imgIdx, 'data':data, 'dataQuick': dataQuick, 'dataQuickP': dataQuickP, 'progress':progress}

#################################
def getDataNamesCTXR(wdir):
    ret=[]
    lstCT=glob.glob('%s/%s*' % (wdir, fileNameInputCT) )
    lstCT.sort()
    if len(lstCT)>0:
        ret.append(lstCT[0])
    else:
        ret.append(None)
    lstXR=glob.glob('%s/%s*' % (wdir, fileNameInputXR_Orig) )
    lstXR.sort()
    if len(lstXR)>0:
        ret.append(lstXR[0])
    else:
        ret.append(None)
    return ret

#################################
def getUniqueDirNameIndex():
    return "userdatadrugres-%s" % datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")

#################################
def printError(ferr, strErr):
    with open(ferr, 'w') as f:
        f.write(strErr)

def setProgress(fprg, parProgress):
    with open(fprg, 'w') as f:
        f.write("%s" % parProgress)

def getProgress(fprg):
    if os.path.isfile(fprg):
        ret=0
        with open(fprg, 'r') as f:
            tmp=f.readline()
            try:
                ret=int(tmp)
            except:
                ret=0
        return ret
    else:
        return -1

def task_proc_drugres(data):
    ptrDirDBXr=data[0]
    ptrDirWdir=data[1]
    pathInpCT,pathInpXR= getDataNamesCTXR(ptrDirWdir)
    pathInpXR_uint8=os.path.join(ptrDirWdir,fileNameInputXR_uint8)
    # X-Ray segmentation:
    pathXRMask    ="%s_maskxr.png" % pathInpXR_uint8
    pathXRMasked  ="%s_maskedxr.png" % pathInpXR_uint8
    pathCTSgmP    ="%s_segmented.png" % pathInpCT
    pathPreviewSgm="%s/preview_segmented.png" % ptrDirWdir
    pathErr="%s/err.txt" % ptrDirDBXr
    pathPrg="%s/progress.txt" % ptrDirWdir
    setProgress(pathPrg, 30)
    retCorr=algxr.task_proc_segmxr2([ptrDirDBXr, pathInpXR_uint8])
    print "XR: retCorr=%s" % retCorr
    if not retCorr:
        printError(pathErr, "Error in X-Ray segmentation: incorrect input image")
        return
    if not os.path.isfile(pathXRMask):
        printError(pathErr, "Error in X-Ray segmentation: unknown error")
        return
    # CT-Segmentation
    setProgress(pathPrg, 70)
    segmCT = algct.SegmentatorCT(ptrDirWdir)
    setProgress(pathPrg, 90)
    segmCT.fnInput=pathInpCT
    segmCT.fnSegmented="%s_maskct.nii.gz"   % os.path.basename(pathInpCT)
    segmCT.fnPreviewInp="%s_previewct.png"  % os.path.basename(pathInpCT)
    segmCT.fnPreviewSgm="%s_segmented.png"  % os.path.basename(pathInpCT)
    segmCT.fnOutputZip="%s_resultct.zip"    % os.path.basename(pathInpCT)
    segmCT.fnError="errct.txt"
    segmCT.segmentCTMask()
    if os.path.isfile("%s/%s" % (ptrDirWdir, segmCT.fnError)):
        printError(pathErr, "Error in CT segmentation: unknown error")
        return
    if not makePreviewForCTXR(pathCTSgmP, pathXRMasked, pathPreviewSgm):
        printError(pathErr, "Unknown in postprocessing stage")
        return
    pathResTXT="%s/res.txt" % ptrDirWdir
    with open(pathResTXT, 'w') as f:
        f.write("Drug Resistatnt\n")
        f.write("93%\n")
        f.write("Not treated before:Drug Resistant:93\n")
        f.write("Treated before:Drug Resistant:83\n")
        f.write("Unknown:Drug Resistant:63\n")
    print "All [Ok]"

class TaskManagerDrugRes:
    def __init__(self, nproc=2):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
    def appendTaskProcessDrugRes(self, parDirDBXr, wdir):
        vdata=[parDirDBXr, wdir]
        # self.pool.apply_async(task_proc_segmxr, [vdata] )
        self.pool.apply_async(task_proc_drugres, [vdata] )

#################################
def getImageInBox(img, smax):
    sizold=img.shape
    siznew=(smax, (smax*sizold[1])/sizold[0])
    posxr=(0, (smax - siznew[1])/2)
    if sizold[1]>sizold[0]:
        siznew=((smax*sizold[0])/sizold[1], smax)
        posxr=((smax - siznew[0])/2, 0)
    imgret=cv2.resize(img, (siznew[1],siznew[0]))
    return (imgret, posxr)

def getPreviewFromCTNifti(pathCTNifti, psiz=(360,180)):
    sizw,sizh=psiz
    datact=nib.load(pathCTNifti).get_data()
    #
    # imgct=cv2.normalize(np.rot90(datact[:,:, datact.shape[2]/2 ], 1), None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
    vMin=-1000.
    vMax=+200.
    imgct=np.rot90(datact[:,:, datact.shape[2]/2 ], 1).astype(np.float)
    imgct=255.*(imgct-vMin)/(vMax-vMin)
    imgct[imgct<0]=0
    imgct[imgct>255.]=255.
    imgct=cv2.resize(imgct.astype(np.uint8), (sizh,sizh))
    return imgct

def makePreviewForCTXR(pathCT, pathXR, pathPV):
    sizw=360
    sizh=180
    if not os.path.isfile(pathXR):
        print "Can't find X-Ray file [%s], skip..." % pathXR
        return False
    if not os.path.isfile(pathCT):
        print "Can't find CT file [%s], skip..." % pathCT
        return False
    imgct1=cv2.imread(pathCT, 1)
    imgxr1=cv2.imread(pathXR, 1)
    imgct,posct=getImageInBox(imgct1, sizh)
    imgxr,posxr=getImageInBox(imgxr1, sizh)
    siz=(sizh, sizw, 3)
    imgout=np.zeros(siz, dtype=np.uint8 )
    imgout[posct[0]:posct[0]+imgct.shape[0], posct[1]:posct[1]+imgct.shape[1], :]=imgct.copy()
    imgout[posxr[0]:posxr[0]+imgxr.shape[0], sizh+posxr[1]:sizh+posxr[1]+imgxr.shape[1], :]=imgxr.copy()
    #
    posX=20
    posY=40
    cv2.putText(imgout,"CT", (posX,posY), cv2.FONT_ITALIC, 0.6, (0,0,0), 3)
    cv2.putText(imgout,"CT", (posX,posY), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)
    cv2.putText(imgout,"X-Ray", (posX+sizh,posY), cv2.FONT_ITALIC, 0.6, (0,0,0), 3)
    cv2.putText(imgout,"X-Ray", (posX+sizh,posY), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)
    #
    cv2.imwrite(pathPV, imgout)
    # print "**** [%s]" % pathPV
    # cv2.imshow("win", imgout)
    # cv2.waitKey(0)
    return True

def makePreviewForDatabase(wdir):
    lstFiles=glob.glob('%s/*.nii.gz' % wdir)
    sizw=360
    sizh=180
    siz=(sizh, sizw)
    for fct in lstFiles:
        fxray='%s_xray.png' % fct
        fprvw='%s_preview.png' % fct
        if not os.path.isfile(fxray):
            print "Can't find X-Ray file [%s], skip..." % fxray
            continue
        # prepare CT-Data
        datact=nib.load(fct).get_data()
        imgct=cv2.normalize(np.rot90(datact[:,:, datact.shape[2]/2 ], 1), None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)
        imgct=cv2.resize(imgct, (sizh,sizh))
        # prepare X-Ray data
        imgxr=cv2.imread(fxray, 0)
        sizxr=imgxr.shape
        newsizxr=(sizh, (sizh*sizxr[1])/sizxr[0])
        posxr=(0, (3*sizh - newsizxr[1])/2)
        if sizxr[1]>sizxr[0]:
            newsizxr=((sizh*sizxr[0])/sizxr[1], sizh)
            posxr=((sizh - newsizxr[0])/2, sizh)
        imgxr=cv2.resize(imgxr, (newsizxr[1],newsizxr[0]))
        # prepare Preview Image
        imgout=np.zeros(siz, dtype=np.uint8 )
        imgout[0:sizh,0:sizh]=imgct.copy()
        imgout[posxr[0]:posxr[0]+newsizxr[0],posxr[1]:posxr[1]+newsizxr[1]]=imgxr.copy()
        posX=20
        posY=40
        cv2.putText(imgout,"CT", (posX,posY), cv2.FONT_ITALIC, 0.6, (0,0,0), 3)
        cv2.putText(imgout,"CT", (posX,posY), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)
        cv2.putText(imgout,"X-Ray", (posX+sizh,posY), cv2.FONT_ITALIC, 0.6, (0,0,0), 3)
        cv2.putText(imgout,"X-Ray", (posX+sizh,posY), cv2.FONT_ITALIC, 0.6, (255,255,255), 2)
        #
        cv2.imwrite(fprvw, imgout)
        print "**** [%s]" % fprvw
        cv2.imshow("win", imgout)
        cv2.waitKey(10)

def postUplodProcessing(parDir):
    fnPreviewIMG=os.path.join(parDir, 'preview.png')
    fnInpCT_Orig=os.path.join(parDir, 'inputct.nii.gz')
    fnInpCT_uint8=os.path.join(parDir, fileNameInputCT_uint8)
    fnInpXR_Orig=glob.glob('%s/inputxrorig.*' % parDir)[0]
    fnInpXR_uint8=os.path.join(parDir, fileNameInputXR_uint8)
    isDicom=False
    try:
        inpDicom=dcm.read_file(fnInpXR_Orig).pixel_array.astype(np.float)
        vmin=inpDicom.min()
        vmax=inpDicom.max()
        imgu8=(255.*(inpDicom-vmin)/(vmax-vmin)).astype(np.uint8)
        imgu8=cv2.equalizeHist(imgu8)
        cv2.imwrite(fnInpXR_uint8, imgu8)
        isDicom=True
    except dcm.errors.InvalidDicomError:
        pass
    if not isDicom:
        imgu8=cv2.imread(fnInpXR_Orig, 0) #cv2.CV_LOAD_IMAGE_GRAYSCALE)
        imgu8=cv2.equalizeHist(imgu8)
        cv2.imwrite(fnInpXR_uint8, imgu8)
    imgCTu8=getPreviewFromCTNifti(fnInpCT_Orig)
    cv2.imwrite(fnInpCT_uint8, imgCTu8)
    makePreviewForCTXR(fnInpCT_uint8,fnInpXR_uint8,fnPreviewIMG)
    ret=os.path.isfile(fnPreviewIMG)  and \
        os.path.isfile(fnInpXR_uint8) and \
        os.path.isfile(fnInpCT_uint8)
    return ret

#################################
def getFileExt(fname):
    tstr=os.path.splitext(fname)
    s1=tstr[1]
    s2=os.path.splitext(tstr[0])[1]
    sext='%s%s' % (s2,s1)
    return sext

#################################
if __name__=="__main__":
    # wdir='/home/ar/big.data/dev/work.django/webcrdf/data/datadb.drugres'
    # makePreviewForDatabase(wdir)
    #
    # dirDBXr="/home/ar/dev-git.git/dev.web/CRDF/webcrdf/data/datadb.segmxr"
    # dirWDir="/home/ar/dev-git.git/dev.web/CRDF/webcrdf/data/users_drugres/yr08tshg5gb9argqsxtp67txqesnkjz0/userdatadrugres-2015_03_16-21_30_29_629969"
    # ##task_proc_drugres( [dirDBXr, dirWDir] )
    # tskMng=TaskManagerDrugRes()
    # tskMng.appendTaskProcessDrugRes(dirDBXr, dirWDir)
    # tskMng.pool.close()
    # tskMng.pool.join()
    #
    # (1) Check Preview generation
    # tdir='/home/ar/github.com/webcrdf.git/webcrdf/data/users_drugres/z0ug90buhwli2n9e09epszullznkuu8o/userdatadrugres-2016_03_28-18_37_26_796649'
    # print postUplodProcessing(tdir)
    #
    # (2) Check DrugResistor Algorithm
    taskManagerDrugRes = TaskManagerDrugRes()
    pathXrDBData='/home/ar/github.com/webcrdf.git/webcrdf/data/datadb.segmxr'
    toDir='/home/ar/github.com/webcrdf.git/webcrdf/data/users_drugres/z0ug90buhwli2n9e09epszullznkuu8o/userdatadrugres-2016_03_28-18_37_26_796649'
    ### taskManagerDrugRes.appendTaskProcessDrugRes(pathXrDBData, toDir)
    task_proc_drugres([pathXrDBData, toDir])
    #
    # pathPCT="/home/ar/big.data/dev/work.django/webcrdf/data/users_drugres/8r74lc0obbcfv65znq1r2u9jaxvwcigf/userdatadrugres-2015_03_12-10_01_44_677575/inputct.nii.gz_segmented.png"
    # pathPXR="/home/ar/big.data/dev/work.django/webcrdf/data/users_drugres/8r74lc0obbcfv65znq1r2u9jaxvwcigf/userdatadrugres-2015_03_12-10_01_44_677575/inputxr.png_maskedxr.png"
    # pathPV="/home/ar/big.data/dev/work.django/webcrdf/data/users_drugres/8r74lc0obbcfv65znq1r2u9jaxvwcigf/userdatadrugres-2015_03_12-10_01_44_677575/preview_segmented.png"
    # makePreviewForCTXR(pathPCT, pathPXR, pathPV)


