#!/usr/bin/env python
__author__ = 'ar'

import math
import time
import os
import sys
import numpy as np
import multiprocessing as mp
import zipfile
import datetime


fileNameInput='input'

#################################
def getUniqueDirNameIndex():
    return "userdata-%s" % datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")

#################################
def task_proc_classify_melanoma(data):
    dirDB =data[0]
    dirScr=data[1]
    imPath=data[2]
    if not os.path.isdir(dirDB):
        return 1
    if not os.path.isdir(dirScr):
        return 2
    if not os.path.isfile(imPath):
        return 3
    mrunStr="cd('%s'); try MelaSearchCom('%s', '%s', '-oi'); catch end; exit" % (dirScr, dirDB, imPath)
    # mrunStr="cd('%s'); MelaSearchCom('%s', '%s', '-oi'); exit" % (dirScr, dirDB, imPath)
    # runStr="matlab2014a -nodesktop -nojvm -r \"%s\" >/dev/null 2>&1" % mrunStr
    runStr="matlab -nodesktop -nojvm -r \"%s\" " % mrunStr
    print "[%s]" % runStr
    os.system(runStr)

class TaskManagerClassifyMelanoma:
    def __init__(self, nproc=2):
        self.nProc  = nproc
        self.pool   = mp.Pool(processes=self.nProc)
    def loadData(self, dirDB, dirScr):
        self.dirDB =dirDB
        self.dirScr=dirScr
    def appendTaskProcessMelanoma(self, fimg):
        vdata=[self.dirDB, self.dirScr, fimg]
        # self.pool.apply_async(task_proc_segmxr, [vdata] )
        self.pool.apply_async(task_proc_classify_melanoma, [vdata] )


#################################
class ResultReader:
    def __init__(self):
        self.nameTEXT = 'out.txt'
        self.nameERR  = 'err.txt'
        self.namePROC = 'out_proc.png'
        self.nameSIM1 = 'out_sim1.png'
        self.nameSIM2 = 'out_sim2.png'
        self.nameSIM3 = 'out_sim3.png'
        self.nameSIM4 = 'out_sim4.png'
    """
    return: (isFinished, ErrCode, ErrStr)
        Errcode '0' -> all is Ok
    """
    def readResultQuick(self, wdir):
        if not os.path.isdir(wdir):
            return (False,1,'Error: input dir does not exist')
        tfErr="%s/%s" % (wdir,self.nameERR)
        if os.path.exists(tfErr):
            f=open(tfErr,'r')
            txtErr = f.readlines()
            f.close()
            return (True,1,txtErr)
        # wdir=os.path.dirname(pathImg)
        fnTEXT='%s/%s' % (wdir, self.nameTEXT)
        fnPROC='%s/%s' % (wdir, self.namePROC)
        if os.path.isfile(fnTEXT) and os.path.isfile(fnPROC):
            return (True,  0, '')
        else:
            return (False, 0, '')
    def readResult(self, wdir, baseURL="/data/"):
        tmpRet=self.readResultQuick(wdir)
        fnTEXT='%s/%s' % (wdir, self.nameTEXT)
        data=[]
        if tmpRet[0] and (tmpRet[1]==0):
            f=open(fnTEXT)
            data=f.readlines()
            f.close()
        imgIdx=os.path.basename(wdir)
        return {'isFinished': tmpRet[0], 'errCode':tmpRet[1], 'errStr':tmpRet[2], 'idx':imgIdx, 'bUrl':baseURL,  'data':data}

#################################
if __name__=="__main__":
    parDirScr='/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma'
    parDirDB ='/home/ar/big.data/dev/work.django/webcrdf/data/datadb.melanomadb'
    tmMelanoma=TaskManagerClassifyMelanoma(nproc=1)
    tmMelanoma.loadData(parDirDB,parDirScr)
    #
    rr=ResultReader()
    print rr.readResultQuick('/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/1')
    print rr.readResultQuick('/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/2')
    print rr.readResultQuick('/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/3')
    #
    print rr.readResult('/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/1')
    #
    parTestImage1='/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/1/004_nevus.jpg'
    parTestImage2='/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/2/03-a-toy-set-sample-002.jpg'
    parTestImage3='/home/ar/big.data/dev/work.django/webcrdf/data/scripts_melanoma/3/06-group01-DA_1.jpg'
    # tmMelanoma.appendTaskProcessMelanoma(parTestImage1)
    # tmMelanoma.appendTaskProcessMelanoma(parTestImage2)
    # tmMelanoma.appendTaskProcessMelanoma(parTestImage3)
    # tmMelanoma.pool.close()
    # tmMelanoma.pool.join()

    # print getUniqueDirNameIndex()

