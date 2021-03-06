# from bzrlib.osutils import isfile
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from django import forms
import alg

from webcrdf.settings import taskManagerSegmCT

import random
import os
import glob
import shutil
import json
import time

# Create your views here.
################################################
def Index(request):
    print '(appsegmct) static URL: %s' % settings.STATIC_ROOT_SEGMCT_USERDATA
    print 'path: [%s]' % request.path
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appsegmct.html', context)
    else:
        context={'next': '/appsegmct/'}
        # return render(request, 'login.html', context)
        return HttpResponseRedirect('/login/?next=%s' % request.path)

################################################
def getInfoAboutImages(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appsegmct:index') )  ##('/')
    sessionId = request.session.session_key
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMCT_USERDATA, sessionId)
    # lstImg=glob.glob('%s/*' % wdir)
    lstDir=glob.glob('%s/*/%s.*' % (wdir, alg.fileNameInput))
    cnt = 0
    ret=[]
    for dd in lstDir:
        ddir=os.path.dirname(dd)
        bnDir=os.path.basename(ddir)
        pathPreviewInp='%s/preview.png' % ddir
        pathPreviewSgm='%s/segmented.png' % ddir
        pathInp=dd
        pathSgm='%s/segmented.nii.gz' % ddir
        pathZip='%s/result.zip' % ddir
        pathErr='%s/err.txt' % ddir
        if os.path.isfile(pathPreviewInp):
            isFinished=False
            if os.path.isfile(pathPreviewSgm):
                isFinished=True
            isError=False
            errorText=""
            if os.path.isfile(pathErr):
                isError=True
                try:
                    f=open(pathErr)
                    errorText=f.read().replace('\n', '<br>')
                    f.close()
                except:
                    errorText='Unknown Error'
            tUrl     = '%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, bnDir, os.path.basename(pathPreviewInp))
            tUrlSegm = '%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, bnDir, os.path.basename(pathPreviewSgm))
            tUrlZip  = '%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, bnDir, os.path.basename(pathZip))
            tUrlInp  = '%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, bnDir, os.path.basename(pathInp))
            tUrlOut = '%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, bnDir, os.path.basename(pathSgm))
            tIdx     = '%s' % bnDir
            timgInfo = {'url': tUrl,
                        'urlSegm':tUrlSegm,
                        'urlZip':tUrlZip,
                        'urlInp':tUrlInp,
                        'urlOut':tUrlOut,
                        'w': 200, 'h': 200,
                        'idx': tIdx,
                        'isFinished': isFinished,
                        'isError':  isError,
                        'txtError': errorText}
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
    if len(settings.IMAGEDB_CT)<num:
        num=len(settings.IMAGEDB_CT)
    ret=[]
    for ii in xrange(0,num):
        tmp=settings.IMAGEDB_CT[ii]
        ret.append(tmp)
    print ret
    return HttpResponse(json.dumps(ret))

################################################
class ImageUplInfo:
    def __init__(self, url, urlSegm, urlInp, urlZip, size, idx, isFinished):
        self.url=url
        self.urlSegm=urlSegm
        self.urlInp=urlInp
        self.urlZip=urlZip
        self.sizeW=size[0]
        self.sizeH=size[1]
        self.idx=idx
        self.isFinished=isFinished

def getUploadedImageList(sessionId):
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMCT_USERDATA, sessionId)
    lstDir=glob.glob('%s/*/%s.*' % (wdir, alg.fileNameInput))
    # lstImg=glob.glob('%s/*' % wdir)
    ret=[]
    cnt = 0
    for dd in lstDir:
        bdir=os.path.dirname(dd)
        pathPreviewInp='%s/preview.png' % bdir
        ddbd=os.path.basename(os.path.dirname(dd))
        if os.path.isfile(pathPreviewInp):
            tfimg='preview.png'
            tmpIsFinished=False
            if os.path.isfile("%s/segmented.png" % bdir):
                tmpIsFinished=True
                tfimg="segmented.png"
            # tIdx     = 'imguser_idx_%05d' % cnt
            tIdx = ddbd
            tmpUrl='%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, ddbd, tfimg)
            tmpUrlInp='%s/users_segmct/%s/%s/%s' % (settings.STATIC_URL, sessionId, ddbd, os.path.basename(dd))
            tmpUrlSegm='%s/users_segmct/%s/%s/segmented.nii.gz' % (settings.STATIC_URL, sessionId, ddbd)
            tmpUrlZip='%s/users_segmct/%s/%s/result.zip' % (settings.STATIC_URL, sessionId, ddbd)
            timgInfo = ImageUplInfo(tmpUrl, tmpUrlSegm, tmpUrlInp, tmpUrlZip, (200,200), tIdx, tmpIsFinished)
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
            widx=alg.getUniqueDirNameIndex()
            odir='%s/%s/%s' % (settings.STATIC_ROOT_SEGMCT_USERDATA, request.session.session_key, widx)
            handle_uploaded_file(odir, fout)
        # return HttpResponseRedirect( reverse('appsegmct:index') )  ##('/')
        return HttpResponse(json.dumps({'strError': 'Error upload file'}))
    else:
        form = UploadFileForm()
    return render_to_response('appsegmct.html', {'form': form})

def UploadFromDB(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        try:
            fname=request.POST['fname']
            fnFrom="%s/%s" % (settings.STATIC_ROOT_SEGMCT_DBDATA,   fname)
            toDir ="%s/%s/%s" % (settings.STATIC_ROOT_SEGMCT_USERDATA, request.session.session_key,alg.getUniqueDirNameIndex())
            if not os.path.isdir(toDir):
                try:
                    os.makedirs(toDir)
                except:
                    print "ERROR: Can't create directory [%s]" % toDir
            fext=alg.getFileExt(fname)
            fnTo  ="%s/%s%s" % (toDir, alg.fileNameInput, fext)
            # fnTo  ="%s/%s" % (toDir, fname)
            print ":: [%s] ---> [%s]" % (fnFrom, fnTo)
            if not os.path.isfile(fnTo):
                if os.path.isfile(fnFrom):
                    shutil.copyfile(fnFrom, fnTo)
                    tmpSegmCT=alg.SegmentatorCT(toDir)
                    if tmpSegmCT.makePreviewInp():
                        pass
                    taskManagerSegmCT.appendTaskSegmCT(toDir)
                    # taskManagerSegmXR.appendTaskSegmXR(fnTo)
        except:
            print "ERROR: Can't copy file from [%s] to [%s]" % (fnFrom, fnTo)
        return HttpResponse(json.dumps([]))
    else:
        return HttpResponseRedirect( reverse('appsegmct:index') )

def handle_uploaded_file(odir, f):
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fext=alg.getFileExt(f.name)
    fout='%s/%s%s' % (odir, alg.fileNameInput, fext)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    tmpSegmCT=alg.SegmentatorCT(odir)
    tmpSegmCT.makePreviewInp()
    # tmpImg=cv2.imread(fout)
    # if tmpImg==None:
    #     print "Error: bad file [%s]" % f
    #     shutil.rmtree(odir)
    #TODO: CHECK POINT
    tmpWDir=os.path.dirname(fout)
    taskManagerSegmCT.appendTaskSegmCT(odir)

def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appsegmct:index') )  ##('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_SEGMCT_USERDATA, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect( reverse('appsegmct:index') ) ##('/')
    shutil.rmtree(wdir)
    return HttpResponseRedirect( reverse('appsegmct:index') ) ##('/')