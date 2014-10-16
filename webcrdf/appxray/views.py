from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django import forms
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from django.conf import settings
from django.conf.urls.static import static

from webcrdf.settings import regXray

import cv2
import numpy as np
import os
import glob
import shutil
import json
import time
import math

def Index(request):
    print 'static URL: %s' % settings.STATIC_ROOT_XRAY_USERDATA
    print 'path: [%s]' % request.path
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appxray.html', context)
    else:
        return render(request, 'login.html')

################################################
class ImageUplInfo:
    def __init__(self, url, size, idx):
        self.url=url
        self.sizeW=size[0]
        self.sizeH=size[1]
        self.idx=idx

def getUploadedImageList(sessionId):
    wdir='%s/%s' % (settings.STATIC_ROOT_XRAY_USERDATA,sessionId)
    lstImg=glob.glob('%s/*' % wdir)
    ret=[]
    for ii in lstImg:
        if os.path.isfile(ii):
            tfimg=os.path.basename(ii)
            timgInfo = ImageUplInfo('%s/users_xray/%s/%s' % (settings.STATIC_URL, sessionId, tfimg), (100,100), 'imguser_%s' % tfimg)
            ret.append( timgInfo )
    return ret

################################################
class UploadFileForm(forms.Form):
    # title = forms.CharField(max_length=50)
    file = forms.FileField()

def Upload(request):
    print '::Upload: session-Id = (%s)' % request.session.session_key
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            odir='%s/%s' % (settings.STATIC_ROOT_XRAY_USERDATA, request.session.session_key)
            handle_uploaded_file(odir, request.FILES['file'])
        return HttpResponseRedirect( reverse('appxray:index') )  ##('/')
    else:
        form = UploadFileForm()
    return render_to_response('appxray.html', {'form': form})

def UploadFromDB(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        try:
            fname=request.POST['fname']
            fnFrom="%s/%s" % (settings.STATIC_ROOT_XRAY_DBDATA,   fname)
            toDir ="%s/%s" % (settings.STATIC_ROOT_XRAY_USERDATA, request.session.session_key)
            if not os.path.isdir(toDir):
                try:
                    os.mkdir(toDir)
                except:
                    print "ERROR: Can't create directory [%s]" % toDir
            fnTo  ="%s/%s" % (toDir, fname)
            print ":: [%s] ---> [%s]" % (fnFrom, fnTo)
            if os.path.isfile(fnFrom):
                shutil.copyfile(fnFrom, fnTo)
        except:
            print "ERROR: Can't copy file from [%s] to [%s]" % (fnFrom, fnTo)
        return HttpResponse(json.dumps([]))
    else:
        return HttpResponseRedirect( reverse('appxray:index') )

def handle_uploaded_file(odir, f):
    if not os.path.isdir(odir):
        os.mkdir(odir)
    fout='%s/%s' % (odir,f.name)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appxray:index') )  ##('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_XRAY_USERDATA, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect( reverse('appxray:index') ) ##('/')
    shutil.rmtree(wdir)
    return HttpResponseRedirect( reverse('appxray:index') ) ##('/')

def getCurrentImageDirRet(request):
    return '%s/%s/ret' % (settings.STATIC_ROOT_XRAY_USERDATA, request.session.session_key)

def loadPTS(fpts):
    pts=np.genfromtxt(fpts, delimiter=',')
    return pts

def calcSizeFromPTS(pts):
    dp12=pts[2,:]-pts[1,:]
    return np.sqrt(np.sum(dp12**2))

def ROISearch(request):
    # time.sleep(2)
    dirRet=getCurrentImageDirRet(request)
    if os.path.isdir(dirRet):
        shutil.rmtree(dirRet)
    if not os.path.isdir(dirRet):
        os.mkdir(dirRet)
    fimgQuery='%s/../%s' % (dirRet, request.POST['idx'])
    ## pre-segment image
    fimgBsn =os.path.basename(fimgQuery)
    ftmpDir="%s/%s_tmp" % (settings.STATIC_ROOT_XRAY_USERDATA, request.session.session_key)
    if not os.path.isdir(ftmpDir):
        os.mkdir(ftmpDir)
    fimgPTS="%s/%s_pts.csv" % (ftmpDir, fimgBsn)
    fimgERR="%s/%s.err" % (ftmpDir, fimgBsn)
    if os.path.isfile(fimgERR):
        ret=json.dumps([1, []])
        return HttpResponse(ret)
    imgPTS=None
    if not os.path.isfile(fimgPTS):
        print "*** Try to register X-Ray image"
        reg=regXray.registerMask(fimgQuery)
        corr=reg[1]
        ptsXY=reg[3]
        if corr<0.7:
            f=open(fimgERR,'w')
            f.write('1')
            f.close()
            ret=json.dumps([1, []])
            return HttpResponse(ret)
        else:
            np.savetxt(fimgPTS, ptsXY, fmt='%0.1f', delimiter=',')
    ptsQuery=loadPTS(fimgPTS)
    # sizReq=calcSizeFromPTS(pts)
    ##
    print 'fuck #2 %s' % fimgQuery
    imgQuery=cv2.imread(fimgQuery, 0) #cv2.CV_LOAD_IMAGE_GRAYSCALE)
    print request.POST
    roi=[int(float(request.POST['x1'])), int(float(request.POST['y1'])), int(float(request.POST['x2'])), int(float(request.POST['y2']))]
    if roi[0]<0:
        roi[0]=0
    if roi[1]<0:
        roi[1]=0
    if roi[2]>=imgQuery.shape[1]:
        roi[2]=imgQuery.shape[1]-1
    if roi[3]>=imgQuery.shape[0]:
        roi[3]=imgQuery.shape[0]-1
    imgQueryP=imgQuery[roi[1]:roi[3], roi[0]:roi[2]].copy()
    (ret_idx_sorted, ret_lst_vcorr, ret_lst_pcorr)=match_list_images2(settings.IMAGEDB_DATA, settings.IMAGEDB_DPTS, imgQueryP, ptsQuery)
    dataPrep=[]
    timeIdx = time.time()
    for ii in xrange(0,10):
        tIdx = ret_idx_sorted[ii]
        tmp=settings.IMAGEDB[tIdx]
        tCorr= ret_lst_vcorr[tIdx]
        pos1=(int(ret_lst_pcorr[tIdx,0]), int(ret_lst_pcorr[tIdx,1]))
        pos2=(int(ret_lst_pcorr[tIdx,2]), int(ret_lst_pcorr[tIdx,3]))
        tmpImg = settings.IMAGEDB_DATA[tIdx].copy()
        p1 = pos1
        p2 = pos2
        cv2.rectangle(tmpImg, p1, p2, (255,255,255), 4, 16) #cv2.CV_AA)
        fimgBaseName='%0.3f_%s' % (timeIdx, os.path.basename(tmp['url']))
        tmpfImg='%s/%s' % (dirRet, fimgBaseName)
        print tmpfImg
        cv2.imwrite(tmpfImg, tmpImg)
        tmp['url2']='/data/users_xray/%s/ret/%s' % (request.session.session_key, fimgBaseName)
        tmp['corr']=tCorr
        tmp['w']=tmpImg.shape[1]
        tmp['h']=tmpImg.shape[0]
        dataPrep.append(tmp)
    ret=json.dumps([0,dataPrep])
    return HttpResponse(ret)

def ImageGallery(request):
    num=10
    try:
        num=request.POST['num']
    except:
        pass
    if len(settings.IMAGEDB)<num:
        num=len(settings.IMAGEDB)
    ret=[]
    for ii in xrange(0,num):
        tmp=settings.IMAGEDB[ii]
        ret.append(tmp)
    return HttpResponse(json.dumps(ret))

#############################################
def match_list_images2(lst_img, lst_pts, imgQuery0, ptsQuery):
    t0=time.time()
    tm_method=cv2.TM_CCOEFF_NORMED
    num_img=len(lst_img)
    ret_lst_vcorr=np.zeros(num_img)
    ret_lst_pcorr=np.zeros((num_img,4))
    # ret_idx_sorted=None
    sizPtsQuery = calcSizeFromPTS(ptsQuery)
    for ii in xrange(0,num_img):
        tmpPts=lst_pts[ii]
        sizPtsTmp = calcSizeFromPTS(tmpPts)
        scaleK = sizPtsTmp/sizPtsQuery
        scaledSize=( int(imgQuery0.shape[1]*scaleK), int(imgQuery0.shape[0]*scaleK) )
        imgQuery=cv2.resize(imgQuery0, scaledSize, interpolation=2)
        tmpCorrMap=cv2.matchTemplate(lst_img[ii], imgQuery, tm_method)
        minv, maxv, minp, maxp = cv2.minMaxLoc(tmpCorrMap)
        ret_lst_vcorr[ii]=maxv
        sizQuery    = imgQuery.shape
        tmpK=1
        maxpP       = (int(maxp[0]*tmpK), int(maxp[1]*tmpK))
        sizQueryP   = (int(sizQuery[0]*tmpK), int(sizQuery[1]*tmpK))
        ret_lst_pcorr[ii,0]=maxpP[0]
        ret_lst_pcorr[ii,1]=maxpP[1]
        ret_lst_pcorr[ii,2]=maxpP[0]+sizQueryP[1]
        ret_lst_pcorr[ii,3]=maxpP[1]+sizQueryP[0]
    ret_idx_sorted=np.argsort(-ret_lst_vcorr)
    print "Match-time: ", (time.time() - t0), "s"
    return (ret_idx_sorted, ret_lst_vcorr, ret_lst_pcorr)

def match_list_images(lst_img, imgQuery):
    t0=time.time()
    tm_method=cv2.TM_CCOEFF_NORMED
    num_img=len(lst_img)
    ret_lst_vcorr=np.zeros(num_img)
    ret_lst_pcorr=np.zeros((num_img,4))
    # ret_idx_sorted=None
    for ii in xrange(0,num_img):
        tmpCorrMap=cv2.matchTemplate(lst_img[ii], imgQuery, tm_method)
        minv, maxv, minp, maxp = cv2.minMaxLoc(tmpCorrMap)
        ret_lst_vcorr[ii]=maxv
        sizQuery    = imgQuery.shape
        tmpK=1
        maxpP       = (int(maxp[0]*tmpK), int(maxp[1]*tmpK))
        sizQueryP   = (int(sizQuery[0]*tmpK), int(sizQuery[1]*tmpK))
        ret_lst_pcorr[ii,0]=maxpP[0]
        ret_lst_pcorr[ii,1]=maxpP[1]
        ret_lst_pcorr[ii,2]=maxpP[0]+sizQueryP[1]
        ret_lst_pcorr[ii,3]=maxpP[1]+sizQueryP[0]
    ret_idx_sorted=np.argsort(-ret_lst_vcorr)
    print "Match-time: ", (time.time() - t0), "s"
    return (ret_idx_sorted, ret_lst_vcorr, ret_lst_pcorr)

