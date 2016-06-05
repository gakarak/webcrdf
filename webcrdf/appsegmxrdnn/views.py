from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from webcrdf.settings import taskManagerSegmXRDNN

from django import forms

import random
import os
import glob
import shutil
import json
import time

################################################
def Index(request):
    print 'static URL: %s' % settings.STATIC_ROOT_SEGMXRDNN_USERDATA
    print 'path: [%s]' % request.path
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appsegmxrdnn.html', context)
    else:
        context={'next': '/appsegmxrdnn/'}
        # return render(request, 'login.html', context)
        return HttpResponseRedirect('/login/?next=%s' % request.path)

################################################
def getInfoAboutImages(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appsegmxrdnn:index') )  ##('/')
    sessionId = request.session.session_key
    ret=getImageListInfoDict(sessionId)
    return HttpResponse(json.dumps(ret))

################################################
def getImageListInfoDict(sessionId):
    wdir = '%s/%s' % (settings.STATIC_ROOT_SEGMXRDNN_USERDATA, sessionId)
    lstDir = glob.glob('%s/*' % wdir)
    cnt = 0
    ret = []
    for dd in lstDir:
        if not os.path.isdir(dd):
            continue
        ddbn = os.path.basename(dd)
        ii = '%s/%s' % (dd, ddbn)
        if os.path.isfile(ii):
            tfimg = os.path.basename(ii)
            isFinished = False
            isGood = True
            tfMask = '%s_maskxr.png' % tfimg
            tfMasked = '%s_maskedxr.png' % tfimg
            tfZip = '%s.zip' % tfimg
            tfErr = 'err.txt'
            tUrl = '%s/users_segmxrdnn/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfimg)
            tUrlZip = '%s/users_segmxrdnn/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfZip)
            if os.path.isfile('%s/%s' % (dd, tfMasked)):
                tUrlSegm = '%s/users_segmxrdnn/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfMasked)
                isFinished = True
            else:
                tUrlSegm = '%s/users_segmxrdnn/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfimg)
            txtError = ''
            tFerrFull = os.path.join(dd, tfErr)
            if os.path.isfile(tFerrFull):
                isGood = False
                isFinished = True
                with open(tFerrFull, 'r') as f:
                    txtError = f.readlines()
            tIdx = 'imguser_idx_%05d' % cnt
            timgInfo = {'url': tUrl, 'urlSegm': tUrlSegm, 'urlZip': tUrlZip,
                        'w': 200, 'h': 200,
                        'idx': tIdx,
                        'isFinished': isFinished, 'isGood': isGood,
                        'textError': txtError}
            ret.append(timgInfo)
            cnt+=1
    return ret

def getImageListInfoAsClass(listImageInfoDict):
    ret=[]
    for ii in listImageInfoDict:
        tmp=ImageUplInfo(ii['url'], (ii['w'],ii['h']), ii['idx'])
        tmp.urlSegm     = ii['urlSegm']
        tmp.urlZip      = ii['urlZip']
        tmp.isFinished  = ii['isFinished']
        tmp.isGood      = ii['isGood']
        tmp.textError   = ii['textError']
        if tmp.isFinished and tmp.isGood:
            tmp.urlInp = tmp.urlSegm
        else:
            tmp.urlInp = tmp.url
        ret.append(tmp)
    return ret

################################################
def ImageGallery(request):
    num=10
    try:
        num=request.POST['num']
    except:
        pass
    if len(settings.IMAGEDB_DNN)<num:
        num=len(settings.IMAGEDB_DNN)
    ret=[]
    for ii in xrange(0,num):
        tmp=settings.IMAGEDB_DNN[ii]
        ret.append(tmp)
    return HttpResponse(json.dumps(ret))

################################################
class ImageUplInfo:
    def __init__(self, url, size, idx):
        self.url=url
        self.sizeW=size[0]
        self.sizeH=size[1]
        self.idx=idx
        self.urlSegm=None
        self.urlZip=None
        self.isFinished=False
        self.isGood=False
        self.textError=''
        self.urlInp = self.url

def getUploadedImageList(sessionId):
    ret = getImageListInfoAsClass(getImageListInfoDict(sessionId))
    return ret

class UploadFileForm(forms.Form):
    # title = forms.CharField(max_length=50)
    file = forms.FileField()

def Upload(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            # odir='%s/%s' % (settings.STATIC_ROOT_SEGMXR_USERDATA, request.session.session_key)
            fout=request.FILES['file']
            odir='%s/%s/%s' % (settings.STATIC_ROOT_SEGMXRDNN_USERDATA, request.session.session_key, fout)
            handle_uploaded_file(odir, fout)
        return HttpResponseRedirect( reverse('appsegmxrdnn:index') )  ##('/')
    else:
        form = UploadFileForm()
    return render_to_response('appsegmxrdnn.html', {'form': form})

def UploadFromDB(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        try:
            fname=request.POST['fname']
            fnFrom="%s/%s" % (settings.STATIC_ROOT_XRAYDNN_DBDATA,   fname)
            toDir ="%s/%s/%s" % (settings.STATIC_ROOT_SEGMXRDNN_USERDATA, request.session.session_key,fname)
            if not os.path.isdir(toDir):
                try:
                    os.makedirs(toDir)
                except:
                    print "ERROR: Can't create directory [%s]" % toDir
            fnTo  ="%s/%s" % (toDir, fname)
            print ":: [%s] ---> [%s]" % (fnFrom, fnTo)
            if not os.path.isfile(fnTo):
                if os.path.isfile(fnFrom):
                    shutil.copyfile(fnFrom, fnTo)
                    #TODO: fix Segmenter Class
                    taskManagerSegmXRDNN.appendTaskSegmXR(fnTo)
                    # taskManagerSegmXR.appendTaskSegmXR(fnTo)
        except Exception as e:
            print("ERROR: [%s] Can't copy file from [%s] to [%s]" % (str(e), fnFrom, fnTo))
        return HttpResponse(json.dumps([]))
    else:
        return HttpResponseRedirect( reverse('appsegmxrdnn:index') )

def handle_uploaded_file(odir, f):
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fout='%s/%s' % (odir, f.name)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    #TODO: fix segmenter class:
    taskManagerSegmXRDNN.appendTaskSegmXR(fout)
    # taskManagerSegmXR.appendTaskSegmXR(fout)

def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appsegmxrdnn:index') )  ##('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMXRDNN_USERDATA, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect( reverse('appsegmxrdnn:index') ) ##('/')
    shutil.rmtree(wdir)
    return HttpResponseRedirect( reverse('appsegmxrdnn:index') ) ##('/')