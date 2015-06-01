from webcrdf.settings import CBIR

__author__ = 'ar'

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django import forms
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from django.views.decorators.cache import cache_page

from django.conf import settings
from django.conf.urls.static import static

import cv2
import numpy as np
import os
import glob
import shutil
import json
import time
import math

def Index(request):
    print 'static URL: %s' % settings.STATIC_ROOT_USERDATA_CBIR
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appcbir.html', context)
    else:
        # return render(request, 'login.html')
        context={'next': '/appcbir/'}
        # return render(request, 'login.html', context)
        return HttpResponseRedirect('/login/?next=%s' % request.path)

def Login(request):
    print '::Login: session-Id = (%s)' % request.session.session_key
    if not 'is_logged' in request.session:
        request.session['is_logged'] = '1'
        request.session['username'] = 'Anonymous'
    return HttpResponseRedirect(reverse('appcbir:index')) ##('/')

# def LoginHtml(request):
#     return render(request, 'login.html')

def Logout(request):
    print '::Logout: session-Id = (%s)' % request.session.session_key
    if 'is_logged' in request.session:
        del request.session['is_logged']
    if 'username' in request.session:
        del request.session['username']
    return HttpResponseRedirect(reverse('appcbir:index')) ##('/')

################################################
@cache_page(60 * 60)
def apiRequestDbInfo(request):
    #ret=json.dumps((CBIR.dataIdxP.tolist(), CBIR.dataPrvPath, CBIR.dataPrvMean, CBIR.dataImgPath))
    ret=json.dumps((CBIR.dataIdxP.tolist(), CBIR.dataPrvPath, CBIR.dataPrvMean))
    # print ret
    return HttpResponse(ret)

################################################
def postProcSearchResults(res):
    imgUrls=[]
    for ii in res[3]:
        imgUrls.append('%s%s/%s.png' % (settings.STATIC_URL, settings.STATIC_DB_DIR, CBIR.dataImgPath[ii]) )
    return (imgUrls, res[1].tolist(), res[0][0:100,9].tolist(), res[3].tolist(), res[0][0:100,8].tolist())

def apiSearchDB(request):
    print request.POST, " : ", len(request.POST)
    if not 'is_logged' in request.session:
        return HttpResponse(json.dumps(()))
    if request.method=='POST':
        qType=json.loads(request.POST['type'])
        res=None
        if qType:
            qFimg=request.POST['fimg']
            # qFimg=json.loads(request.POST['fimg'])
            fimg = '%s/%s/%s' % (settings.STATIC_ROOT_USERDATA_CBIR, request.session.session_key, qFimg)
            print fimg
            res=CBIR.findNgbh(fimg, numRet=10)
            # res=CBIR.findNgbh2(fimg, numRet=10)
            # res=CBIR.findNgbhV(fimg, numRet=10)
            # return HttpResponse(json.dumps(()))
        else:
            idxImg = int(request.POST['idx'])
            # idxImgPath="%s/%s.png" % (settings.STATIC_ROOT_DATADB, CBIR.dataImgPath[idxImg])
            # print "[%d] : %s" % (idxImg, idxImgPath)
            res=CBIR.findNgbhInDB(idxImg,numRet=10)
            # res=CBIR.findNgbhInDB2(idxImg,numRet=10)
            # res=CBIR.findNgbhInDBV(idxImg,numRet=10)
        ret=postProcSearchResults(res)
        # 0: urls, 1: dist, 2: CT-index, 3: Img-index, 4: Img-Index in CT
        return HttpResponse(json.dumps(ret))
    else:
        return HttpResponse(json.dumps(()))

################################################
class ImageUplInfo:
    def __init__(self, url, size, idx):
        self.url=url
        self.sizeW=size[0]
        self.sizeH=size[1]
        self.idx=idx

def getUploadedImageList(sessionId):
    wdir='%s/%s' % (settings.STATIC_ROOT_USERDATA_CBIR,sessionId)
    lstImg=glob.glob('%s/*' % wdir)
    print lstImg
    ret=[]
    for ii in lstImg:
        if os.path.isfile(ii):
            tfimg=os.path.basename(ii)
            timgInfo = ImageUplInfo('%s/users_cbir/%s/%s' % (settings.STATIC_URL, sessionId, tfimg), (100,100), 'imguser_%s' % tfimg)
            ret.append( timgInfo )
    return ret

################################################
def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_USERDATA_CBIR, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect(reverse('appcbir:index'))
    shutil.rmtree(wdir)
    return HttpResponseRedirect(reverse('appcbir:index'))


################################################
class UploadFileForm(forms.Form):
    # title = forms.CharField(max_length=50)
    file = forms.FileField()

def Upload(request):
    print '::Upload: session-Id = (%s)' % request.session.session_key
    if not 'is_logged' in request.session:
        return HttpResponseRedirect(reverse('ray:index')) ##('/')
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            odir='%s/%s' % (settings.STATIC_ROOT_USERDATA_CBIR, request.session.session_key)
            handle_uploaded_file(odir, request.FILES['file'])
        return HttpResponseRedirect(reverse('appcbir:index')) ##('/')
    else:
        form = UploadFileForm()
        # return HttpResponseRedirect('/')
    # return render_to_response('upload.html', {'form': form})
    return render_to_response('index.html', {'form': form})

def handle_uploaded_file(odir, f):
    if not os.path.isdir(odir):
        os.mkdir(odir)
    fout='%s/%s' % (odir,f.name)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

################################################
