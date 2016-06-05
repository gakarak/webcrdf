#!/usr/bin/env python
__author__ = 'ar'

import math
import time
import os
import sys
import numpy as np
import cv2
import multiprocessing as mp
import zipfile
import dicom as dcm

import matplotlib.pyplot as plt
from algext import textSegNetInference

#################################
def task_proc_segmxr_DNN(data):
    # ptrPathWdir     = data[0]
    ptrPathCaffe    = data[0]
    ptrPathWeights  = data[1]
    ptrPathImg      = data[2]
    ptrPathWdir     = os.path.dirname(ptrPathImg)
    segmXRdnn       = SegmentatorXRayDNN(parCaffeRoot=ptrPathCaffe)
    tret = segmXRdnn.runSergmentation(ptrPathImg, ptrPathWdir, ptrPathWeights)
    if tret:
        segmXRdnn.saveMasksToFiles(ptrPathImg)
        fzip="%s.zip" % ptrPathImg
        zObj=zipfile.ZipFile(fzip, 'w')
        zipDir='%s_dir' % os.path.basename(ptrPathImg)
        lstFimg=segmXRdnn.lstFnToZip
        for ff in lstFimg:
            ffbn = os.path.basename(ff)
            zObj.write(ff, "%s/%s" % (zipDir, ffbn))
        return True
    return False

def task_proc_segmxr_DNN_EXEC(data):
    strCMD="python %s %s %s %s" % (os.path.abspath(__file__), data[0], data[1], data[2])
    if os.system(strCMD)==0:
        return True
    return False

class TaskManagerSegmXR_DNN:
    def __init__(self, nproc=1):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
    def setParams(self, pathCaffe, pathModelDNN):
        self.pathCaffe  = pathCaffe
        self.pathModel  = pathModelDNN
    def appendTaskSegmXR(self, fimg):
        vdata=[self.pathCaffe, self.pathModel, fimg]
        # task_proc_segmxr_DNN(vdata)
        # task_proc_segmxr_DNN_EXEC(vdata)
        # self.pool.apply_async(task_proc_segmxr_DNN, [vdata] )
        self.pool.apply_async(task_proc_segmxr_DNN_EXEC, [vdata])

#################################
class SegmentatorXRayDNN:
    sizeImage   = (480,360)
    caffeRoot   = None
    idxFileName = 'idx-tmp.txt'
    protoTxtInference = 'inference.txt'
    msk=None
    imgProc=None
    imgMasked=None
    mskOnImg=None
    wdir=None
    fnErrdDef='err.txt'
    def __init__(self, parCaffeRoot = None):
        self.textInference = textSegNetInference
        self.caffeRoot = parCaffeRoot
    def runSergmentation(self, pathToImage, wdir, pathWeights):
        self.wdir = wdir
        if not self.checkFilePath(pathToImage):
            return False
        if not self.checkFilePath(pathWeights):
            return False
        if not self.checkDirectory(self.wdir):
            return False
        if not self.checkDirectory(self.caffeRoot):
            return False
        #FIXME: remove extra-check code
        if os.path.isfile(pathToImage) and os.path.isfile(pathWeights) and os.path.isdir(self.wdir):
            isDicom = False
            isLoadedImage=False
            try:
                inpDicom = dcm.read_file(pathToImage).pixel_array.astype(np.float)
                vmin = inpDicom.min()
                vmax = inpDicom.max()
                imgu80 = (255. * (inpDicom - vmin) / (vmax - vmin)).astype(np.uint8)
                # imgu8 = cv2.equalizeHist(imgu80)
                isDicom = True
                isLoadedImage = True
            except dcm.errors.InvalidDicomError:
                pass
            if not isDicom:
                try:
                    imgu80 = cv2.imread(pathToImage, 0)  # cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    # imgu8 = cv2.equalizeHist(imgu80)
                    if imgu80 is not None:
                        isLoadedImage = True
                except Exception:
                    pass
            if not isLoadedImage:
                self.logError('Cant load image [%s]' % pathToImage)
                return False
            # FIXME: chack this point
            imgu8 = cv2.equalizeHist(imgu80)
            # imgu8 = imgu80
            if len(imgu8.shape)<3:
                imgu8 = cv2.cvtColor(imgu8,cv2.COLOR_GRAY2BGR)
            inputSizeImg = (imgu80.shape[1],imgu80.shape[0])
            imgu8 = cv2.resize(imgu8, self.sizeImage, interpolation=cv2.INTER_CUBIC)
            foutImg = os.path.join(self.wdir, os.path.basename(os.path.splitext(pathToImage)[0]) + '-proc.jpg')
            self.imgProc = imgu8
            self.fimgProc = foutImg
            cv2.imwrite(foutImg, imgu8)
            tmpIdxFilename = os.path.join(self.wdir,self.idxFileName)
            with open(tmpIdxFilename,'w') as f:
                f.write(foutImg)
                f.write(" ")
                f.write(foutImg)
                f.write("\n")
            tmpProtoInfFilename = os.path.join(self.wdir, self.protoTxtInference)
            with open(tmpProtoInfFilename, 'w') as f:
                f.write(self.textInference % tmpIdxFilename)
            print('-----')
            try:
                sys.path.insert(0, os.path.join(self.caffeRoot, "python"))
                sys.path.append("/usr/local/cuda/lib64")
                import caffe
                caffe.set_mode_cpu()
                net = caffe.Net(tmpProtoInfFilename, pathWeights, caffe.TEST)
                net.forward()
                predicted = net.blobs['prob'].data
                output = np.squeeze(predicted[0, :, :, :])
                # free memory:
                del net
            except Exception as e:
                self.logError('Error when processing Caffe-Net (%s)' % str(e))
                return False
            ind = np.argmax(output, axis=0)
            r = ind.copy()
            g = ind.copy()
            b = ind.copy()
            self.msk = 255 * np.uint8((r + g + b) > 0)
            self.msk = cv2.resize(self.msk, inputSizeImg, interpolation=cv2.INTER_CUBIC)
            if self.msk is not None:
                self.imgMasked  = self.makeMaskedImage(imgu80, self.msk)
                self.mskOnImg   = self.makeImgOnMask(imgu80, self.msk)
            return True
    def saveMasksToFiles(self, pathImg):
        self.checkDirectory(self.wdir)
        foutImg=os.path.join(self.wdir, os.path.basename(pathImg))
        pathImgMask = "%s_maskxr.png" % foutImg
        pathImgMasked = "%s_maskedxr.png" % foutImg
        pathImgOnMask = "%s_onmaskxr.png" % foutImg
        self.pathImg        = foutImg
        self.pathImgMask    = pathImgMask
        self.pathImgMasked  = pathImgMasked
        self.pathImgOnMask  = pathImgOnMask
        try:
            cv2.imwrite(pathImgMask,   self.msk)
            cv2.imwrite(pathImgMasked, self.imgMasked)
            cv2.imwrite(pathImgOnMask, self.mskOnImg)
            #
            self.lstFnToZip=(self.pathImg,
                        self.pathImgMask,
                        self.pathImgMasked,
                        self.pathImgOnMask)
            return True
        except Exception as e:
            self.logError('Cant save output images [%s]' % str(e))
        return False
    def logError(self, strError):
        if not os.path.isdir(self.wdir):
            raise Exception("Cant find <working> directory [%s]" % self.wdir)
        fnError = os.path.join(self.wdir, self.fnErrdDef)
        strErrorFinal='%0.5f : %s' % (time.time(), strError)
        print (strErrorFinal)
        with open(fnError, 'wa') as f:
            f.write('%s\n' % strErrorFinal)
    def checkDirectory(self, pathDir):
        if not os.path.isdir(pathDir):
            self.logError('Cant find directory [%s]' % pathDir)
            return False
        return True
    def checkFilePath(self, pathToFile):
        if not os.path.isfile(pathToFile):
            self.logError('Cant find file [%s]' % pathToFile)
            return False
        return True
    def makeMaskedImage(self, imgInp, msk):
        img = imgInp
        if (len(imgInp.shape)<3):
            img = cv2.cvtColor(imgInp, cv2.COLOR_GRAY2BGR)
        tmp = img[:, :, 2]
        tmp[msk > 0] = 255
        img[:, :, 2] = tmp
        return img
    def makeImgOnMask(self, img, msk):
        if (len(img.shape)>2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (len(msk.shape)>2):
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        tmpImg = img.astype(np.int32)
        tmpImg = tmpImg + 1
        tmpImg[tmpImg > 255] = 255
        tmpImg[msk == 0] = 0
        tmpImg = tmpImg.astype(np.uint8)
        return tmpImg

#################################
def test_main0():
    lstFn = [
        "/home/ar/tmp/test_xray_dicom_0/1_1.dcm",
        "/home/ar/tmp/test_xray_dicom_1/1_1.dcm",
        "/home/ar/tmp/test_xray_dicom_2/2_1.dcm",
        "/home/ar/tmp/test_xray_dicom_3/1_1.dcm",
        "/home/ar/tmp/test_xray_dicom_4/1_1.dcm",
        "/home/ar/tmp/test_xray_dicom_5/1_1.png",
        "/home/ar/tmp/test_xray_dicom_6/1_1.png"]
    pathXRay = '/home/ar/github.com/webcrdf.git/webcrdf/data/users_segmxrdnn/rqb75teiu16y5xpin9f648fctxsci8ft/033_000115E0_an.dcm.png/033_000115E0_an.dcm.png'
    pathModel = '/home/ar/github.com/webcrdf.git/webcrdf/data/scripts_segmxrdnn/segnet_xray_weights.caffemodel'
    pathCaffe = '/home/ar/github.com/webcrdf.git/webcrdf/data/scripts_segmxrdnn/caffe-segnet.git-build'
    tdata = [pathCaffe, pathModel, pathXRay]
    task_proc_segmxr_DNN(tdata)
    #
    # pathModel = '/home/ar/github.com/webcrdf.git/webcrdf/data/scripts_segmxrdnn/segnet_xray_weights.caffemodel'
    # pathCaffe  = '/home/ar/deep-learning/caffe-segnet.git-build'
    # taskManager = TaskManagerSegmXR_DNN()
    # taskManager.setParams(pathCaffe=pathCaffe, pathModelDNN=pathModel)
    # for ii in lstFn:
    #     taskManager.appendTaskSegmXR(ii)
    # taskManager.pool.close()
    # taskManager.pool.join()

#################################
def usage(argv):
    print('Usage: %s {/path/to/caffe-dir} {/path/to/weights.caffemodel} {/path/to/image.[dcm,jpg,png,...]}'
          % os.path.basename(argv[0]))

#################################
if __name__=="__main__":
    if len(sys.argv)<4:
        usage(sys.argv)
        sys.exit(1)
    pathCaffe = sys.argv[1]
    pathModel = sys.argv[2]
    pathImage = sys.argv[3]
    tdata = [pathCaffe, pathModel, pathImage]
    if not task_proc_segmxr_DNN(tdata):
        sys.exit(1)
