from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from webcrdf.settings import taskManagerSegmXR

from django import forms

import random
import os
import glob
import shutil
import json
import time

# Create your views here.
################################################
def Index(request):
    print 'static URL: %s' % settings.STATIC_ROOT_XRAY_USERDATA
    print 'path: [%s]' % request.path
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appsegmxr.html', context)
    else:
        context={'next': '/appsegmxr/'}
        # return render(request, 'login.html', context)
        return HttpResponseRedirect('/login/?next=%s' % request.path)

################################################
def getInfoAboutImages(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appsegmxr:index') )  ##('/')
    sessionId = request.session.session_key
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMXR_USERDATA, sessionId)
    # lstImg=glob.glob('%s/*' % wdir)
    lstDir=glob.glob('%s/*' % wdir)
    cnt = 0
    ret=[]
    for dd in lstDir:
        if not os.path.isdir(dd):
            continue
        ddbn=os.path.basename(dd)
        ii='%s/%s' % (dd, ddbn)
        if os.path.isfile(ii):
            tfimg=os.path.basename(ii)
            isFinished=False
            isGood=True
            tfMask='%s_mask.png' % tfimg
            tfMasked='%s_masked.png' % tfimg
            tfZip='%s.zip' % tfimg
            tfErr='%s.err' % tfimg
            tUrl    = '%s/users_segmxr/%s/%s' % (settings.STATIC_URL, sessionId, tfimg)
            tUrlZip =  '%s/users_segmxr/%s/%s' % (settings.STATIC_URL, sessionId, tfZip)
            if os.path.isfile('%s/%s' % (dd, tfMasked)):
                tUrlSegm = '%s/users_segmxr/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfMasked)
                isFinished=True
            else:
                tUrlSegm = '%s/users_segmxr/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfimg)
            if os.path.isfile('%s/%s' % (dd, tfErr)):
                isGood = False
            tIdx     = 'imguser_idx_%05d' % cnt
            timgInfo = {'url': tUrl, 'urlSegm':tUrlSegm, 'urlZip':tUrlZip, 'w': 200, 'h': 200, 'idx': tIdx, 'isFinished': isFinished, 'isGood': isGood}
            ret.append( timgInfo )
        cnt +=1
    return HttpResponse(json.dumps(ret))

################################################
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

################################################
class ImageUplInfo:
    def __init__(self, url, size, idx):
        self.url=url
        self.sizeW=size[0]
        self.sizeH=size[1]
        self.idx=idx

def getUploadedImageList(sessionId):
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMXR_USERDATA, sessionId)
    lstDir=glob.glob('%s/*' % wdir)
    # lstImg=glob.glob('%s/*' % wdir)
    ret=[]
    cnt = 0
    for dd in lstDir:
        if not os.path.isdir(dd):
            continue
        ddbn=os.path.basename(dd)
        ii='%s/%s' % (dd, ddbn)
        if os.path.isfile(ii):
            tfimg=os.path.basename(ii)
            # tIdx     = 'imguser_%s' % tfimg
            tIdx     = 'imguser_idx_%05d' % cnt
            timgInfo = ImageUplInfo('%s/users_segmxr/%s/%s/%s' % (settings.STATIC_URL, sessionId, tfimg, tfimg), (200,200), tIdx)
            ret.append( timgInfo )
        cnt+=1
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
            odir='%s/%s/%s' % (settings.STATIC_ROOT_SEGMXR_USERDATA, request.session.session_key, fout)
            handle_uploaded_file(odir, fout)
        return HttpResponseRedirect( reverse('appsegmxr:index') )  ##('/')
    else:
        form = UploadFileForm()
    return render_to_response('appsegmxr.html', {'form': form})

def UploadFromDB(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        try:
            fname=request.POST['fname']
            fnFrom="%s/%s" % (settings.STATIC_ROOT_XRAY_DBDATA,   fname)
            toDir ="%s/%s/%s" % (settings.STATIC_ROOT_SEGMXR_USERDATA, request.session.session_key,fname)
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
                    taskManagerSegmXR.appendTaskSegmXR(fnTo)
        except:
            print "ERROR: Can't copy file from [%s] to [%s]" % (fnFrom, fnTo)
        return HttpResponse(json.dumps([]))
    else:
        return HttpResponseRedirect( reverse('appxray:index') )

def handle_uploaded_file(odir, f):
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fout='%s/%s' % (odir, f.name)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    taskManagerSegmXR.appendTaskSegmXR(fout)

def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appsegmxr:index') )  ##('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMXR_USERDATA, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect( reverse('appsegmxr:index') ) ##('/')
    shutil.rmtree(wdir)
    return HttpResponseRedirect( reverse('appsegmxr:index') ) ##('/')