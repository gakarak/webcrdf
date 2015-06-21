	#!/usr/bin/env python
import skimage

__author__ = 'ar'

import numpy as np
from skimage.feature import greycomatrix
import sys
import sklearn.metrics as MT
import array
# import skimage.color
# import skimage.io
# import skimage.transform
import cv2
from skimage.filter import threshold_otsu
from skimage import io
from skimage import draw
from skimage import exposure
import os
import struct
import math

import matplotlib.pyplot as plt

import scipy
import scipy.ndimage as ndi


###############################################
def getCircMask(rad):
    y,x = np.ogrid[-rad :rad+1, -rad :rad+1]
    msk=( (x**2 + y**2)<=rad**2)
    return msk

def getMask(img):
    timgb=(timg>10)
    se=getCircMask(5)
    timgm=ndi.binary_dilation(ndi.binary_erosion(timgb, structure=se), structure=se)
    msk=np.zeros(timg.shape, dtype=np.bool)
    timgLBL,numLBL = ndi.label(timgm>0)
    print numLBL
    if numLBL>0:
        sums = ndi.sum(timgm, labels=timgLBL, index=range(1,numLBL+1))
        print sums
        idxMax=np.argsort(-sums)[0]+1
        msk[timgLBL==idxMax]=True
    return msk

###############################################
def readBinData(fname, ptype=np.int64):
    f=open(fname,'r')
    siz=array.array('i')
    siz.read(f,2)
    siz=(siz[1],siz[0])
    data=np.fromfile(f,dtype=ptype)
    data=data.reshape(siz)
    f.close()
    return data

def writeBinData(fname, data, ptype=np.int64):
    f=open(fname,'wb')
    siz=array.array('i')
    siz.append(data.shape[1])
    siz.append(data.shape[0])
    siz.write(f)
    data.reshape(-1).astype(ptype).tofile(f)
    f.close()

def calcDSC_IID135(img, nbit=3):
    bshift=8-nbit
    pimg=(img>>bshift)
    res = greycomatrix(pimg, [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=(1<<nbit), symmetric=True)
    dsc=np.reshape(np.sum(res,3).astype(np.float),-1)
    return dsc/np.sum(dsc)

def calcDSC_IID135_D2(img, nbit=3):
    bshift=8-nbit
    pimg=(img>>bshift)
    res = greycomatrix(pimg, [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=(1<<nbit), symmetric=True)
    res = np.sum(res,3).astype(np.float64)
    idxDiag=np.tri(res.shape[0],res.shape[1])
    dsc0=res[:,:,0]
    dsc1=res[:,:,1]
    dsc2=res[:,:,2]
    dsc=np.concatenate( [dsc0[idxDiag>0], dsc1[idxDiag>0], dsc2[idxDiag>0]] )
    return dsc/np.sum(dsc)

def calcDSC_IID135_G2(img,nbit=3):
    bshift=8-nbit
    pimg=(img>>bshift)
    siz=pimg.shape
    pp=(siz[0]/2,siz[1]/2)
    res00 = greycomatrix(pimg[:pp[0],:pp[1]], [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=(1<<nbit), symmetric=True)
    res01 = greycomatrix(pimg[:pp[0],pp[1]:], [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=(1<<nbit), symmetric=True)
    res10 = greycomatrix(pimg[pp[0]:,:pp[1]], [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=(1<<nbit), symmetric=True)
    res11 = greycomatrix(pimg[pp[0]:,pp[1]:], [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=(1<<nbit), symmetric=True)
    dsc00 = np.sum(res00,3).astype(np.float)
    dsc01 = np.sum(res01,3).astype(np.float)
    dsc10 = np.sum(res10,3).astype(np.float)
    dsc11 = np.sum(res11,3).astype(np.float)
    idx2 = np.tri(dsc00.shape[0],dsc00.shape[1])
    idx3 = np.zeros( (idx2.shape[0],idx2.shape[1],3) )
    idx3[:,:,0]=idx2
    idx3[:,:,1]=idx2
    idx3[:,:,2]=idx2
    dsc = np.concatenate( [dsc00[idx3>0], dsc01[idx3>0], dsc10[idx3>0], dsc11[idx3>0]] )
    return dsc/np.sum(dsc)

def calcDSC_IID135_Mask(img, msk, nbit=3):
    bshift=8-nbit
    pimg=(img>>bshift)
    maxIdx=(1<<nbit)
    numLevels=maxIdx+1
    pimg[msk>0]=maxIdx
    res=greycomatrix(pimg, [1,3,5], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=numLevels, symmetric=True)
    res=np.sum(res,3).astype(np.float64)
    res[maxIdx,maxIdx,:]=0
    nrm=img.shape[0]*img.shape[1]*4
    idxDiag=np.tri(res.shape[0],res.shape[1])
    dsc0=res[:,:,0]
    dsc1=res[:,:,1]
    dsc2=res[:,:,2]
    dsc=np.concatenate( [dsc0[idxDiag>0], dsc1[idxDiag>0], dsc2[idxDiag>0]] )
    return dsc/nrm

############################################
class FBank:
    def __init__(self):
        self.fbank=None
        self.isLoaded=False
        self.filterCh=1
        self.numSigms=0
        self.filterSize=None
        self.filters=None
    def loadFBank(self, fbank):
        if os.path.isfile(fbank):
            self.fbank=fbank
            self.isLoaded=False
            f=open(self.fbank, 'rb')
            self.filterCh=struct.unpack('<I', f.read(4))[0]
            self.numSigms=struct.unpack('<I', f.read(4))[0]
            self.filterSize={}
            self.filters={}
            for ii in xrange(0,self.numSigms):
                tmpFSiz=struct.unpack('<I', f.read(4))[0]
                tmpSigm=struct.unpack('<I', f.read(4))[0]
                tmpNumF =struct.unpack('<I', f.read(4))[0]
                tmpLstFlt=[]
                # print "(%d) siz=%d, sigm=%d, numf=%d" % (ii, tmpFSiz, tmpSigm, tmpNumF)
                buffSiz=tmpFSiz*tmpFSiz*self.filterCh*8
                for ff in xrange(0,tmpNumF):
                    buff=f.read(buffSiz)
                    flt=np.frombuffer(buff, np.float64)
                    if self.filterCh==1:
                        flt=flt.reshape( (tmpFSiz, tmpFSiz) )
                    else:
                        flt=flt.reshape( (tmpFSiz, tmpFSiz, self.filterCh))
                    tmpLstFlt.append(flt)
                self.filterSize[tmpSigm]=tmpFSiz
                self.filters[tmpSigm]=tmpLstFlt
            f.close()
            self.isLoaded=True
            print self.toString()
        else:
            self.isLoaded=False
            print "*** Error: Can't find FilterBank file [%s]" % fbank
    def toString(self):
        ret='fbank( not loaded )={}'
        if self.isLoaded:
            ret='fbank(nch=%d, nsgm=%d)={' % (self.filterCh, self.numSigms)
            for kk in self.filters.keys():
                val=self.filters[kk]
                ret += '(%d, numFlt=%d, fltSiz=%s), ' % (kk, len(val), val[0].shape)
            ret += '}'
        return ret
    def showFilters(self):
        for ss in self.filters.keys():
            val=self.filters[ss]
            numc    = int(math.floor(math.sqrt(len(val))))
            numr    = int(math.ceil(len(val)/numc))
            f,ax=plt.subplots(numc,numr)
            f.suptitle('sigma=%d' % ss)
            cnt=0
            for cc in xrange(0,numc):
                for rr in xrange(0,numr):
                    if self.filterCh==1:
                        ax[cc,rr].imshow(val[cnt])
                    elif self.filterCh==2:
                        ax[cc,rr].imshow(val[cnt][:,:,0])
                    elif self.filterCh==3:
                        tmp=cv2.normalize(val[cnt], None, 0,255, 32, cv2.CV_8U) #cv2.cv.CV_MINMAX=32=cv2.NORM_MINMAX
                        ax[cc,rr].imshow(tmp)
                    else:
                        tmp=cv2.normalize(val[cnt][:,:,0:3], None, 0,255, 32, cv2.CV_8U) #cv2.cv.CV_MINMAX=32=cv2.NORM_MINMAX
                        ax[cc,rr].imshow(tmp)
                    cnt+=1
        plt.show()
    def filterImageR(self, img, flt):
        ret=cv2.filter2D(img, cv2.CV_64F, flt) #.astype(np.float64)
        if len(ret.shape)==2:
            return ret
        else:
            return np.sum(ret,2)
    def filterImageBankFltR(self, img, sgmKey, fltIdx):
        if self.isLoaded:
            flt=self.filters[sgmKey][fltIdx]
            ret=self.filterImageR(img,flt)
        else:
            ret=None
        return ret
    def filterBinImageFromBankR(self, img, sgmKey, fltIdx):
        return (self.filterImageBankFltR(img, sgmKey,fltIdx)>0)
    def filterBinImageR(self, img, flt):
        return (self.filterImageR(img, flt)>0).astype(np.int32)
    #
    def filterImage(self, img, flt):
        ret=cv2.filter2D(img, cv2.CV_64F, flt) #.astype(np.float64)
        return ret
    def filterImageBankFlt(self, img, sgmKey, fltIdx):
        if self.isLoaded:
            flt=self.filters[sgmKey][fltIdx]
            ret=self.filterImage(img,flt)
        else:
            ret=None
        return ret
    def calcDscBFPCAForSigm(self, img, sgm, numf):
        flts=self.filters[sgm]
        tmp=self.filterBinImageR(img,flts[0])
        for ii in xrange(1,numf):
           tmp+=(self.filterBinImageR(img,flts[ii])<<ii)
        bins=range(0, 1+(1<<numf))
        ret=np.histogram(tmp, bins=bins)[0]
        return ret
    """
    img - numpy image
    sigmCodes - dict like {sigm1:numFlt1, sigm2:numFlt2, ...}
    """
    def calcDscBFPCA(self, img, sigmCodes):
        ret=None
        if not self.isLoaded:
            return ret
        for ssi in sigmCodes.keys():
            numfi=sigmCodes[ssi]
            if ret==None:
                ret=self.calcDscBFPCAForSigm(img, ssi, numfi)
            else:
                tmp=self.calcDscBFPCAForSigm(img, ssi, numfi)
                ret=np.concatenate([ret, tmp])
            # print "%d->%d" % (ssi, numfi)
        ret=ret.astype(np.float64)/np.sum(ret)
        return ret
    def calcDscBFPCAForFile(self, fimg, sigmCodes):
        if self.isLoaded:
            if self.filterCh==1:
                img=cv2.imread(fimg, 0)
            else:
                img=cv2.imread(fimg, 1)
            return self.calcDscBFPCA(img, sigmCodes)
        else:
            return None

############################################
def calcDscGridHist(img, ngrid=3, nbit=3):
    bshift=8-nbit
    pimg=(img>>bshift)
    bins=range(0,1+(1<<nbit))
    ret=None
    dr=img.shape[0]/ngrid
    dc=img.shape[1]/ngrid
    for rr in xrange(0,ngrid):
        r0=rr*dr
        r1=r0+dr
        for cc in xrange(0,ngrid):
            c0=cc*dc
            c1=c0+dc
            tmp=np.histogram(pimg[r0:r1, c0:c1], bins=bins)[0]
            if ret==None:
                ret=tmp
            else:
                ret=np.concatenate([ret, tmp])
    ret=ret.astype(np.float64)
    ret=ret/np.sum(ret)
    return ret

############################################
def getLines(fname):
    ret=None
    with open(fname,'r') as f:
        ret=f.read().splitlines()
    return ret

############################################
def preprocImageV(fimg):
    nbin=64
    dn = 24
    img=skimage.img_as_float(io.imread(fimg,as_grey=True))
    siz=img.shape
    sizMax=np.max(siz)
    siz2=(sizMax,sizMax)
    msk=np.zeros(siz)
    cv2.circle(msk, (siz[1]/2, siz[0]/2), (np.min(siz)/2)-3, (1,1,1), -1) ##cv2.cv.CV_FILLED
    s0 = 0.3440
    h1,xx = exposure.histogram(img[msk==1], nbin)
    s1 = np.std(img[msk==1])
    h  = h1.copy()
    h[np.argwhere(h > 0)[-1]] = 0
    h[np.argwhere(h > 0)[-1]] = 0
    st = np.percentile(img[msk==1], 1)
    p1 = np.argwhere(np.fabs(xx - st)==np.min(abs(xx - st)))[0][0]
    h[p1:np.min((nbin, p1 + np.round(dn * s1 / s0)))]=0
    p2 = np.argwhere(h==np.max(h))[0][0]
    max1=xx[p1]
    max2=xx[p2]
    ret= 0.8*(img-max1)/(max2-max1)
    ret[ret<0]=0
    ret[ret>1]=1
    ret=np.uint8(255*ret)
    ret2=np.zeros(siz2, np.uint8)
    r0=np.floor((sizMax-siz[0])/2)
    c0=np.floor((sizMax-siz[1])/2)
    ret2[r0:r0+siz[0], c0:c0+siz[1]]=ret
    # cv2.imshow("ret", ret)
    # cv2.imshow("ret2", ret2)
    # cv2.waitKey(0)
    return ret2

def preprocImage(fimg):
    nbin=64
    img=io.imread(fimg,as_grey=True)
    siz=img.shape
    msk=np.zeros(siz, np.uint8)
    rr,cc=draw.circle(siz[1]/2, siz[0]/2, (np.min(siz)/2)-3)
    msk[rr,cc]=1
    thresh=threshold_otsu(img[msk==1])
    msk1=msk&(img<=thresh)
    msk2=msk&(img>thresh)
    ret1=exposure.histogram(img[msk1==1],nbins=nbin)
    ret2=exposure.histogram(img[msk2==1],nbins=nbin)
    var1=0*np.std(img[msk1==1])
    max1=-var1+ret1[1][np.argmax(ret1[0])]
    max2=ret2[1][np.argmax(ret2[0])]
    ret=(0.8*(img-max1)/(max2-max1))
    ret[ret<0]=0
    ret[ret>1]=1
    # print np.min(ret), " * " , np.max(ret)
    return np.uint8(255*ret)
    # return ret

############################################
class SCBIR:
    def __init__(self, wdir):
        self.wdir=wdir
        self.fnIdx='%s/../index_db_images.bin' % self.wdir
        # self.fnDsc='%s/../data_dsc_iid135_var.bin' % self.wdir
        # self.fnDsc='%s/../data_dsc_iid135_pca.bin' % self.wdir
        # self.fnDsc='%s/../data_dsc_iid135.bin' % self.wdir
        self.fnDsc='%s/../data_dsc_ghist_3b3g.bin' % self.wdir
        # self.fnDsc='%s/../data_dsc_iid135_msk2.bin' % self.wdir
        self.fnPCA='%s/../data_dsc_iid135_mat_pca.bin' % self.wdir
        ##
        self.fnDsc2='%s/../data_dsc_iid135_g2_92.bin' % self.wdir
        self.fnDsc2Idx='%s/../data_dsc_iid135_g2_92.idx' % self.wdir
        # self.fnDsc2='%s/../data_dsc_iid135_g2.bin' % self.wdir
        # self.fnDsc2Idx='%s/../data_dsc_iid135_g2_92.idx' % self.wdir
        ##
        self.fnDscV='%s/../data_dsc_iid135_92.bin' % self.wdir
        self.fnDscVIdx='%s/../data_dsc_iid135_92.idx' % self.wdir
        self.numIdx=64
        ##
        self.fnImgPath='%s/../index_db_images_fn.csv' % self.wdir
        self.fnPrvPath='%s/../index_db_preview_fn.csv' % self.wdir
        # self.fnPrvMean='%s/../index_db_preview_mean2.csv' % self.wdir
        # self.fnPrvMean='%s/../index_db_preview_var2.csv' % self.wdir
        # self.fnPrvMean='%s/../index_db_preview_vlung2.csv' % self.wdir
        # self.fnPrvMean='%s/../index_db_preview_vbody2.csv' % self.wdir
        # self.fnPrvMean='%s/../index_db_preview_lung2body.csv' % self.wdir
        self.fnPrvMean='%s/../index_db_preview_numslices.csv' % self.wdir
        self.fnIdxPreview='%s/../index_db_preview.bin' % self.wdir
    def load(self):
        self.dataIdx=readBinData(self.fnIdx, np.int64)
        self.dataIdxP=readBinData(self.fnIdxPreview, np.int64)
        self.dataImgPath=getLines(self.fnImgPath)
        self.dataPrvPath=getLines(self.fnPrvPath)
        ## Prepare Mean-Preview
        #self.dataPrvMean=getLines(self.fnPrvMean)
        tmpData=np.genfromtxt(self.fnPrvMean, delimiter=',')
        tmpData=tmpData/np.max(tmpData)
        pr=np.percentile(tmpData,(2,98))
        tmpData=64*(tmpData-pr[0])/(pr[1]-pr[0])
        tmpData=tmpData.astype(np.int32)
        tmpData[tmpData<0]=0
        tmpData[tmpData>63]=63
        tmpDataList=tmpData.tolist()
        # print tmpDataList
        self.dataPrvMean=tmpDataList
        # print data.tolist()
        ### DSC-MSK
        self.dataDsc=readBinData(self.fnDsc, np.float64)
        self.dataPCA=readBinData(self.fnPCA, np.float64)
        ### DSC-PCA
        # self.dataDsc=readBinData(self.fnDsc, np.float64)
        # self.dataPCA=readBinData(self.fnPCA, np.float64)
        ### DSC-G2
        # self.dataDsc2Idx=(readBinData(self.fnDsc2Idx, np.float64).astype(np.int)-1)[0,:]
        # self.dataDsc2=readBinData(self.fnDsc2, np.float64)
        #
        ### DSC-V
        # self.dataDscVIdx=(readBinData(self.fnDscVIdx, np.float64).astype(np.int)-1)[0,:]
        # self.dataDscV=readBinData(self.fnDscV, np.float64)
        #
        # self.dataDscVIdx=self.dataDscVIdx[0,:self.numIdx]
        # self.dataDscV=self.dataDscV[:,self.dataDscVIdx]
    def printInfo(self):
        print "-- Info --"
        print "Wdir:   ", self.wdir
        print "Idx:    ", self.dataIdx.shape
        print "IdxP:   ", self.dataIdxP.shape
        try:
            str1="Dsc-PCA:    ", self.dataDsc.shape
            str2="Idx-PCA:    ", self.dataPCA.shape
            print str1
            print str2
        except:
            pass
        try:
            str1="Dsc2:   ", self.dataDsc2.shape
            str2="Dsc2Idx:", self.dataDsc2Idx.shape
            print str1
            print str2
        except:
            pass
        try:
            str1="DscV:   ", self.dataDscV.shape
            str2="DscVIdx:", self.dataDscVIdx.shape
            print str1
            print str2
        except:
            pass
        print "ImgPath: ", len(self.dataImgPath)
        print "PrvPath: ", len(self.dataPrvPath)
    """
    PCA-dsc
    """
    def getDSCPCA(self, fnImg):
        timg=preprocImageV(fnImg)
        timg=cv2.resize(timg,(256,256))
        # tdsc = calcDSC_IID135(timg,3)
        tdsc = calcDscGridHist(timg, nbit=3, ngrid=3)
        # tdscPCA = np.transpose(self.dataPCA).dot(tdsc)
        tdscPCA=tdsc
        return tdscPCA
    def findNgbh(self, fimg, numRet=5):
        dscPCA=self.getDSCPCA(fimg)
        tdst=MT.pairwise_distances(self.dataDsc, dscPCA, metric='l1')[:,0]
        # tdst=MT.pairwise_distances(self.dataDsc[:, :1], dscPCA[:1], metric='l1')[:,0]
        dstSortIdx=np.argsort(tdst)[:numRet]
        print dstSortIdx
        lstFn=[]
        for ii in dstSortIdx:
            tmpFImage='%s/%s.png' % (self.wdir, self.dataImgPath[ii])
            lstFn.append(tmpFImage)
        return (self.dataIdx[dstSortIdx,:], tdst[dstSortIdx], lstFn, dstSortIdx)
    def findNgbhInDB(self, idxImg, numRet=5):
        dscPCA = self.dataDsc[idxImg,:]
        # tdst = MT.pairwise_distances(self.dataDsc[:, :64], dscPCA[:64], metric='l1')[:,0]
        # tdst = MT.pairwise_distances(self.dataDsc[:, 2::3], dscPCA[2::3], metric='l1')[:,0]
        tdst = MT.pairwise_distances(self.dataDsc, dscPCA, metric='l1')[:,0]
        idxIdent=0 # user-ID
        tid=self.dataIdx[idxImg,idxIdent]
        arrUID=self.dataIdx[:,idxIdent]
        tdst[arrUID==tid]=100.
        dstSortIdx=np.argsort(tdst)[:numRet]
        # print "tID: ", tid
        lstFn=[]
        for ii in dstSortIdx:
            tmpFImage='%s/%s.png' % (self.wdir, self.dataImgPath[ii])
            lstFn.append(tmpFImage)
        return (self.dataIdx[dstSortIdx,:], tdst[dstSortIdx], lstFn, dstSortIdx)
    """
    G2-dsc + variance index
    """
    def getDSCPCA2(self, fnImg):
        timg=preprocImageV(fnImg)
        timg=cv2.resize(timg,(256,256))
        tdsc = calcDSC_IID135_G2(timg,3)
        tdscIDX = tdsc[self.dataDsc2Idx]
        return tdscIDX
    def findNgbh2(self, fimg, numRet=5):
        dscPCA2=self.getDSCPCA2(fimg)
        tdst=MT.pairwise_distances(self.dataDsc2, dscPCA2, metric='l1')[:,0]
        dstSortIdx=np.argsort(tdst)[:numRet]
        print "findNgbh2(): " , dstSortIdx
        lstFn=[]
        for ii in dstSortIdx:
            tmpFImage='%s/%s.png' % (self.wdir, self.dataImgPath[ii])
            lstFn.append(tmpFImage)
        return (self.dataIdx[dstSortIdx,:], tdst[dstSortIdx], lstFn, dstSortIdx)
    def findNgbhInDB2(self, idxImg, numRet=5):
        dscPCA2 = self.dataDsc2[idxImg,:]
        tdst = MT.pairwise_distances(self.dataDsc2, dscPCA2, metric='l1')[:,0]
        idxIdent=0 # user-ID
        tid=self.dataIdx[idxImg,idxIdent]
        arrUID=self.dataIdx[:,idxIdent]
        # tdst[arrUID==tid]=100.
        dstSortIdx=np.argsort(tdst)[:numRet]
        lstFn=[]
        for ii in dstSortIdx:
            tmpFImage='%s/%s.png' % (self.wdir, self.dataImgPath[ii])
            lstFn.append(tmpFImage)
        return (self.dataIdx[dstSortIdx,:], tdst[dstSortIdx], lstFn, dstSortIdx)
    """
    Variance Index
    """
    def getDSCPCAV(self, fnImg):
        timg=preprocImageV(fnImg)
        timg=cv2.resize(timg,(256,256))
        tdsc = calcDSC_IID135(timg,3)
        tdscIDX = tdsc[self.dataDscVIdx]
        print "tdscPCA.shape = ", tdscIDX.shape
        return tdscIDX
    def findNgbhV(self, fimg, numRet=5):
        dscPCAV=self.getDSCPCAV(fimg)
        tdst=MT.pairwise_distances(self.dataDscV, dscPCAV, metric='l1')[:,0]
        dstSortIdx=np.argsort(tdst)[:numRet]
        print "findNgbhV(): " , dstSortIdx
        lstFn=[]
        for ii in dstSortIdx:
            tmpFImage='%s/%s.png' % (self.wdir, self.dataImgPath[ii])
            lstFn.append(tmpFImage)
        return (self.dataIdx[dstSortIdx,:], tdst[dstSortIdx], lstFn, dstSortIdx)
    def findNgbhInDBV(self, idxImg, numRet=5):
        dscPCAV = self.dataDscV[idxImg,:]
        tdst = MT.pairwise_distances(self.dataDscV, dscPCAV, metric='l1')[:,0]
        idxIdent=0 # user-ID
        tid=self.dataIdx[idxImg,idxIdent]
        arrUID=self.dataIdx[:,idxIdent]
        tdst[arrUID==tid]=100.
        dstSortIdx=np.argsort(tdst)[:numRet]
        lstFn=[]
        for ii in dstSortIdx:
            tmpFImage='%s/%s.png' % (self.wdir, self.dataImgPath[ii])
            lstFn.append(tmpFImage)
        return (self.dataIdx[dstSortIdx,:], tdst[dstSortIdx], lstFn, dstSortIdx)
    def findNgbhInDB3(self, idxCT, idxSlice, numRet=5):
        idx=self.calcLinerIndex(idxCT, idxSlice)
        print "Idx= ", idx
        return self.findNgbhInDB(idx,numRet)
    def calcLinerIndex(self, idxCT, idxSlice):
        # print "Fuck: ", self.dataIdxP[idxCT, :]
        return self.dataIdxP[idxCT, 6]+idxSlice

def_wdir='/home/ar/big.data/data.CRDF-CT/dbdir_ct_10k'
# def_fimg='/home/ar/big.data/data.CRDF-CT/dbdir_ct_10k/F00_00544/F00_00544_20060811_2_000.png'
def_fimg='/home/ar/big.data/data.CRDF-CT/dbdir_ct_10k/F00_00544/F00_00544_20060811_2_022.png'


# import matplotlib.pyplot as plt

############################################
if __name__=="__main__":
    fdata='/home/ar/big.data/data.CRDF-CT/index_db_preview_numslices.csv'
    data=np.genfromtxt(fdata, delimiter=',')
    data=data/np.max(data)
    pr=np.percentile(data,(20,97))
    data=64*(data-pr[0])/(pr[1]-pr[0])
    data=data.astype(np.int32)
    data[data<0]=0
    data[data>63]=63
    print data.tolist()

    sys.exit(0)
    # fnames='/home/ar/MEGA/data/CRDF/test_with_masks/images.txt'
    # fnames='/home/ar/MEGA/data/CRDF/images.txt'
    fnames='/home/ar/MEGA/data/CRDF/test2/images.txt'
    if not os.path.isfile(fnames):
        print "Error: can't found imagelist-file [%s]" % fnames
        sys.exit(1)
    lstfimg=getLines(fnames)
    numf=len(lstfimg)
    cnt=0
    fdsc='%s_dsc.csv' % fnames
    dataDsc=None
    for ff in lstfimg:
        fmsk='%s_mask.png' % ff
        if not os.path.isfile(ff):
            print "Error: can't find image-file [%s]" % ff
            sys.exit(1)
        if not os.path.isfile(fmsk):
            print "Error: can't find mask-file [%s]" % fmsk
            sys.exit(1)
        timg=cv2.imread(ff, 0) #cv2.CV_LOAD_IMAGE_GRAYSCALE)
        tmsk=cv2.imread(fmsk, 0)
        tdsc=calcDSC_IID135_Mask(timg, tmsk)
        # tdsc=calcDSC_IID135_D2(timg)
        if dataDsc==None:
            dataDsc=np.zeros( (numf,len(tdsc)), np.float64)
        dataDsc[cnt,:]=tdsc
        cnt+=1
        # f,xarr = plt.subplots(1,3)
        # xarr[0].imshow(timg)
        # xarr[1].imshow(tmsk)
        # xarr[2].imshow(getMask(timg))
        # plt.show()
    # np.savetxt(fdsc, dataDsc, fmt='%.18e', delimiter=',')
    sys.exit(1)
    ##############################
    cbir=SCBIR(def_wdir)
    cbir.load()
    cbir.printInfo()
    plt.get_cmap()
    ##############################

    # fimg='/home/ar/img/lena.png'
    # fimg='/home/ar/big.data/data.CRDF-CT/clinical_records_20140823_181334.unzip/131/CT/20120227/2_107.dcm.jpg'
    # qq=preprocImageV(fimg)
    ## qq=np.uint8(255*io.imread('/home/ar/img/lena.png', as_grey=True))
    # plt.imshow(qq, cmap=plt.cm.gray)
    # print np.min(qq), " * " , np.max(qq)
    # plt.show()
#    idxQuery=85
#    ret=cbir.findNgbhInDB(idxQuery)
#    fig, arr = plt.subplots(1,6)
#    arr[0].imshow(plt.imread('%s/%s.png' % (cbir.wdir,cbir.dataImgPath[idxQuery]) ))
#    print cbir.dataDsc[idxQuery,:]
#    for ii in xrange(0,5):
#        # print ret[2][ii]
#        tmp=(255*plt.imread(ret[2][ii] )).astype(np.uint8)
#        print np.max(tmp)
#        print cbir.dataDsc[ret[3][ii],:]
#        arr[ii+1].imshow( ((tmp>>5)<<5) )
#        arr[ii+1].set_title('%d : dst=%0.4f' % (ret[3][ii], ret[1][ii]) )
#    plt.show()
