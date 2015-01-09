# from bzrlib.osutils import isfile
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from webcrdf.settings import taskManagerClassMelanoma
import alg

from django import forms

import random
import os
import glob
import shutil
import json
import time

import cv2

# Create your views here.
################################################
def Index(request):
    print '*MELANOMA* static URL: %s' % settings.STATIC_ROOT_XRAY_USERDATA
    print 'path: [%s]' % request.path
    print request.COOKIES
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appmelanoma.html', context)
    else:
        return render(request, 'login.html')

################################################
def getInfoAboutImages(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appmelanoma:index') )  ##('/')
    sessionId = request.session.session_key
    wdir='%s/%s' % (settings.STATIC_ROOT_MELANOMA_USERDATA, sessionId)
    # lstImg=glob.glob('%s/*' % wdir)
    lstDir=glob.glob('%s/*' % wdir)
    cnt = 0
    ret=[]
    rr=alg.ResultReader()
    for dd in lstDir:
        if not os.path.isdir(dd):
            continue
        tUrl    = '/%s/%s/%s' % (settings.URL_MELANOMA_USERDATA, sessionId, os.path.basename(dd))
        tmpRes = rr.readResult(dd, tUrl)
        ret.append( tmpRes )
        cnt +=1
    return HttpResponse(json.dumps(ret))

################################################
def ImageGallery(request):
    num=10
    try:
        num=request.POST['num']
    except:
        pass
    if len(settings.IMAGEDB_MELANOMA)<num:
        num=len(settings.IMAGEDB_MELANOMA)
    ret=[]
    for ii in xrange(0,num):
        tmp=settings.IMAGEDB_MELANOMA[ii]
        ret.append(tmp)
    return HttpResponse(json.dumps(ret))

################################################
class ImageUplInfo:
    def __init__(self, url, urlSegm, isSegm, size, idx):
        self.url=url
        self.urlSegm=urlSegm
        self.isSegm=isSegm
        self.sizeW=size[0]
        self.sizeH=size[1]
        self.idx=idx

def getMinRescaledSize(siz, maxSize):
    if (siz[0]>1) and (siz[1]>1):
        siz0=maxSize
        siz1=int(maxSize*float(siz[1])/float(siz[0]))
        if siz[0]>siz[1]:
            siz1=maxSize
            siz0=int(maxSize*float(siz[0])/float(siz[1]))
        return (siz0,siz1)
    else:
        return (maxSize,maxSize)


def getUploadedImageList(sessionId):
    wdir='%s/%s' % (settings.STATIC_ROOT_MELANOMA_USERDATA, sessionId)
    # lstDir=glob.glob('%s/*' % wdir)
    lstImg=glob.glob('%s/*/%s.*' % (wdir,alg.fileNameInput))
    ret=[]
    cnt = 0
    rr=alg.ResultReader()
    for ii in lstImg:
        if not os.path.isfile(ii):
            continue
        timg=cv2.imread(ii,0)
        if timg==None:
            continue
        siz=getMinRescaledSize((timg.shape[1], timg.shape[0]), 100)
        # ddbn=os.path.basename(dd)
        # ii='%s/%s' % (dd, ddbn)
        # if os.path.isfile(ii):
        tfimg=os.path.basename(ii)
        tdir =os.path.basename(os.path.dirname(ii))
        # tIdx     = 'imguser_%s' % tfimg
        # tIdx     = 'imguser_idx_%05d' % cnt
        tIdx = tdir
        tmpURL ='%s/users_melanoma/%s/%s/%s' % (settings.STATIC_URL, sessionId, tdir, tfimg)
        tmpURLS='%s/users_melanoma/%s/%s/out_proc.png' % (settings.STATIC_URL, sessionId, tdir)
        tmpIsSegm=False
        tmpInfo=rr.readResultQuick(os.path.dirname(ii))
        if(tmpInfo[0] and (tmpInfo[1]==0)):
            tmpIsSegm=True
        timgInfo = ImageUplInfo(tmpURL, tmpURLS, tmpIsSegm, (siz[0],siz[1]), tIdx)
        print tmpInfo
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
        widx=None
        if form.is_valid():
            # odir='%s/%s' % (settings.STATIC_ROOT_SEGMXR_USERDATA, request.session.session_key)
            fout=request.FILES['file']
            # odir='%s/%s/%s' % (settings.STATIC_ROOT_MELANOMA_USERDATA, request.session.session_key, fout)
            widx=alg.getUniqueDirNameIndex()
            odir='%s/%s/%s' % (settings.STATIC_ROOT_MELANOMA_USERDATA, request.session.session_key, widx)
            handle_uploaded_file(odir, fout)
        ret = HttpResponseRedirect( reverse('appmelanoma:index') )
        ret.set_cookie('widx', widx)
        return  ret ##('/')
    else:
        form = UploadFileForm()
    return render_to_response('appmelanoma.html', {'form': form})

def UploadFromDB(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        widx=None
        try:
            fname=request.POST['fname']
            fnFrom="%s/%s" % (settings.STATIC_ROOT_MELANOMA_DBDATA,   fname)
            toDir ="%s/%s/%s" % (settings.STATIC_ROOT_MELANOMA_USERDATA, request.session.session_key, alg.getUniqueDirNameIndex())
            if not os.path.isdir(toDir):
                try:
                    os.makedirs(toDir)
                except:
                    print "ERROR: Can't create directory [%s]" % toDir
            fext=os.path.splitext(fname)[1]
            fnTo  ="%s/%s%s" % (toDir, alg.fileNameInput, fext)
            print ":: [%s] ---> [%s]" % (fnFrom, fnTo)
            if not os.path.isfile(fnTo):
                if os.path.isfile(fnFrom):
                    shutil.copyfile(fnFrom, fnTo)
                    #TODO: CHECK POINT
                    tmpWDir=os.path.dirname(fnTo)
                    widx=os.path.basename(tmpWDir)
                    taskManagerClassMelanoma.appendTaskProcessMelanoma(fnTo)
                    # taskManagerSegmXR.appendTaskSegmXR(fnTo)
        except:
            print "ERROR: Can't copy file from [%s] to [%s]" % (fnFrom, fnTo)
        ret=HttpResponse(json.dumps({'widx': widx}))
        ret.set_cookie('widx', widx)
        return ret
    else:
        return HttpResponseRedirect( reverse('appmelanoma:index') )

def handle_uploaded_file(odir, f):
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fext=os.path.splitext(f.name)[1]
    fout='%s/%s%s' % (odir, alg.fileNameInput, fext)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    tmpImg=cv2.imread(fout)
    if tmpImg==None:
        print "Error: bad file [%s]" % f
        shutil.rmtree(odir)
    #TODO: CHECK POINT
    tmpWDir=os.path.dirname(fout)
    taskManagerClassMelanoma.appendTaskProcessMelanoma(fout)
    # taskManagerSegmXR.appendTaskSegmXR(fout)

def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appmelanoma:index') )  ##('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_MELANOMA_USERDATA, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect( reverse('appmelanoma:index') ) ##('/')
    shutil.rmtree(wdir)
    ret=HttpResponseRedirect( reverse('appmelanoma:index') )
    ret.delete_cookie('widx')
    return  ret ##('/')

if __name__=='__main__':
    print alg.fileNameInput
