# from bzrlib.osutils import isfile
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

from webcrdf.settings import taskManagerDrugRes
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
    print '*DRUGRES* static URL: %s' % settings.STATIC_ROOT_DRUGRES_USERDATA
    print 'path: [%s]' % request.path
    print request.COOKIES
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        uploadedImages=getUploadedImageList(request.session.session_key)
        context={'userName': userName, 'uploadedImages': uploadedImages}
        return render(request, 'appdrugres.html', context)
    else:
        context={'next': '/appdrugres/'}
        # return render(request, 'login.html', context)
        return HttpResponseRedirect('/login/?next=%s' % request.path)

################################################
def getInfoAboutImages(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appdrugres:index') )  ##('/')
    sessionId = request.session.session_key
    uploadedImages=getUploadedImageList(request.session.session_key)
    ret=[]
    for ii in uploadedImages:
        ret.append(ii.toJSon())
    # wdir='%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, sessionId)
    # lstDir=glob.glob('%s/*' % wdir)
    # cnt = 0
    # ret=[]
    # rr=alg.ResultReader()
    # for dd in lstDir:
    #     if not os.path.isdir(dd):
    #         continue
    #     if not rr.checkWDir(dd):
    #         continue
    #     qr=rr.readResultQuick(dd)
    #     tdir=os.path.basename(dd)
    #     tmpUrl      ='%s/users_drugres/%s/%s/preview.png'                   % (settings.STATIC_URL, sessionId, tdir)
    #     tmpUrlSegm  ='%s/users_drugres/%s/%s/preview_segmented.png'         % (settings.STATIC_URL, sessionId, tdir)
    #     tmpUrlCT    ='%s/users_drugres/%s/%s/inputct.nii.gz'                 % (settings.STATIC_URL, sessionId, tdir)
    #     tmpUrlCTSegm='%s/users_drugres/%s/%s/inputct.nii.gz_maskct.nii.gz'   % (settings.STATIC_URL, sessionId, tdir)
    #     tmpUrlZip   ='%s/users_drugres/%s/%s/inputct.nii.gz_resultct.zip'    % (settings.STATIC_URL, sessionId, tdir)
    #     tIdx     = tdir
    #     isFinished  =qr[0]
    #     isError     =qr[1]
    #     errorText   =qr[2]
    #     progress    =qr[3]
    #     timgInfo = {'url': tmpUrl,
    #                     'urlSegm'   :tmpUrlSegm,
    #                     'urlZip'    :tmpUrlZip,
    #                     'urlCT'     :tmpUrlCT,
    #                     'urlCTSegm' :tmpUrlCTSegm,
    #                     'w': 200, 'h': 200,
    #                     'idx': tIdx,
    #                     'isFinished': isFinished,
    #                     'isError':  isError,
    #                     'txtError': errorText,
    #                     'progress': progress}
    #     ret.append( timgInfo )
    #     cnt +=1
    return HttpResponse(json.dumps(ret))

################################################
def ImageGallery(request):
    num=10
    try:
        num=request.POST['num']
    except:
        pass
    if len(settings.IMAGEDB_CT_DRUGRES)<num:
        num=len(settings.IMAGEDB_CT_DRUGRES)
    ret=[]
    for ii in xrange(0,num):
        tmp=settings.IMAGEDB_CT_DRUGRES[ii]
        ret.append(tmp)
    return HttpResponse(json.dumps(ret))

################################################
class ImageUplInfo:
    def __init__(self):
        self.url=None
        self.urlSegm=None
        self.urlCT=None
        self.urlCTSegm=None
        self.urlZip=None
        self.idx=0
        self.isFinished=False
        self.isError=False
        self.errStr=""
        self.progress=0
        self.data=None
        self.dataQuick=None
        self.dataQuickP=0
    def loadImageInfo(self, wdir):
        tdir        = os.path.basename(wdir)
        sessionId   = os.path.basename(os.path.dirname(wdir))
        rr=alg.ResultReader()
        res=rr.readResult(wdir)
        self.idx        = tdir
        self.isFinished = res['isFinished']
        self.isError    = res['errCode']
        self.txtError   = res['errStr']
        self.dataQuick  = res['dataQuick']
        self.dataQuickP = res['dataQuickP']
        self.data       = res['data']
        self.progress   = res['progress']
        #
        self.url        = '%s/users_drugres/%s/%s/preview.png' % (settings.STATIC_URL, sessionId, tdir)
        self.urlSegm    = '%s/users_drugres/%s/%s/preview_segmented.png' % (settings.STATIC_URL, sessionId, tdir)
        self.urlCT      = '%s/users_drugres/%s/%s/inputct.nii.gz' % (settings.STATIC_URL, sessionId, tdir)
        self.urlCTSegm  = '%s/users_drugres/%s/%s/inputct.nii.gz_maskct.nii.gz' % (settings.STATIC_URL, sessionId, tdir)
        self.urlZip     = '%s/users_drugres/%s/%s/inputct.nii.gz_resultct.zip' % (settings.STATIC_URL, sessionId, tdir)
    def toJSon(self):
        ret={
            'url'       : self.url,
            'urlSegm'   : self.urlSegm,
            'urlZip'    : self.urlZip,
            'urlCT'     : self.urlCT,
            'urlCTSegm' : self.urlCTSegm,
            'idx'       : self.idx,
            'isFinished': self.isFinished,
            'isError'   : self.isError,
            'txtError'  : self.txtError,
            'progress'  : self.progress,
            'dataQuick' : self.dataQuick,
            'dataQuickP': self.dataQuickP,
            'data'      : self.data
        }
        print ret
        return ret

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
    wdir='%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, sessionId)
    lstDir=glob.glob('%s/*' % wdir)
    ret=[]
    cnt = 0
    rr=alg.ResultReader()
    for ii in lstDir:
        if not os.path.isdir(ii):
            continue
        if not rr.checkWDir(ii):
            continue
        # qr=rr.readResultQuick(ii)
        # isFinished  = qr[0]
        # siz=(256,256)
        # tfimg="%s/preview.png" % os.path.basename(ii)
        # tdir =os.path.basename(ii)
        # tIdx = tdir
        # tmpUrl      ='%s/users_drugres/%s/%s/preview.png'                   % (settings.STATIC_URL, sessionId, tdir)
        # tmpUrlSegm  ='%s/users_drugres/%s/%s/preview_segmented.png'         % (settings.STATIC_URL, sessionId, tdir)
        # tmpUrlCT    ='%s/users_drugres/%s/%s/inputct.nii.gz'                 % (settings.STATIC_URL, sessionId, tdir)
        # tmpUrlCTSegm='%s/users_drugres/%s/%s/inputct.nii.gz_maskct.nii.gz'   % (settings.STATIC_URL, sessionId, tdir)
        # tmpUrlZip   ='%s/users_drugres/%s/%s/inputct.nii.gz_resultct.zip'    % (settings.STATIC_URL, sessionId, tdir)
        # timgInfo = ImageUplInfo(tmpUrl,tmpUrlSegm, tmpUrlCT, tmpUrlCTSegm, tmpUrlZip,
        #                         (siz[0],siz[1]), tIdx, isFinished)
        timgInfo=ImageUplInfo()
        timgInfo.loadImageInfo(ii)
        ret.append( timgInfo )
        cnt+=1
    return ret

class UploadFileForm(forms.Form):
    # title = forms.CharField(max_length=50)
    file = forms.FileField()

def UploadCT(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            fout=request.FILES['file']
            # widx=alg.getUniqueDirNameIndex()
            # odir='%s/%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key, widx)
            odir='%s/%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key, settings.SDRUGRES_TMPDIR)
            handle_uploaded_file_ct(odir, fout)
            return HttpResponse(json.dumps( {"err": False} ))
        else:
            return HttpResponse(json.dumps( {"err": True} ))
    else:
        form = UploadFileForm()
    return render_to_response('appdrugres.html', {'form': form})

def UploadXR(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            fout=request.FILES['file']
            odir='%s/%s/tmp_data' % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key)
            handle_uploaded_file_xr(odir, fout)
            return HttpResponse(json.dumps( {"err": False} ))
        else:
            return HttpResponse(json.dumps( {"err": True} ))
    else:
        form = UploadFileForm()
    return render_to_response('appdrugres.html', {'form': form})

def UploadFromDB(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect('/')
    if request.method == 'POST':
        widx=None
        try:
            fname=request.POST['fname']
            fnFromCT="%s/%s" % (settings.STATIC_ROOT_DRUGRES_DBDATA,   fname)
            fnFromXR="%s/%s_xray.png" % (settings.STATIC_ROOT_DRUGRES_DBDATA,       fname)
            fnFromPV="%s/%s_preview.png" % (settings.STATIC_ROOT_DRUGRES_DBDATA,    fname)
            if not (os.path.isfile(fnFromCT) and os.path.isfile(fnFromXR) and os.path.isfile(fnFromPV)):
                return HttpResponseRedirect( reverse('appdrugres:index') )
            toDir ="%s/%s/%s" % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key, alg.getUniqueDirNameIndex())
            if not os.path.isdir(toDir):
                try:
                    os.makedirs(toDir)
                except:
                    print "ERROR: Can't create directory [%s]" % toDir
            fnToCT="%s/inputct.nii.gz" % toDir
            fnToXR="%s/inputxrorig.png" % toDir
            fnToXR_uint8="%s/inputxr_uint8.png" % toDir
            fnToPV="%s/preview.png" % toDir
            # fext=os.path.splitext(fname)[1]
            # fnTo  ="%s/%s%s" % (toDir, alg.fileNameInputCT, fext)
            print ":: CT: [%s] ---> [%s]" % (fnFromCT, fnToCT)
            shutil.copyfile(fnFromCT, fnToCT)
            print ":: XR: [%s] ---> [%s]" % (fnFromXR, fnToXR)
            shutil.copyfile(fnFromXR, fnToXR)
            print ":: XR: [%s] ---> [%s]" % (fnFromXR, fnToXR_uint8)
            shutil.copyfile(fnFromXR, fnToXR_uint8)
            print ":: PV: [%s] ---> [%s]" % (fnFromPV, fnToPV)
            shutil.copyfile(fnFromPV, fnToPV)
            #TODO: CHECK POINT
            taskManagerDrugRes.appendTaskProcessDrugRes(settings.STATIC_ROOT_SEGMXR_DBDATA, toDir)
        except:
            print "ERROR: Can't copy file from Database"
        ret=HttpResponse(json.dumps({'widx': widx}))
        # ret.set_cookie('widx', widx)
        return ret
    else:
        return HttpResponseRedirect( reverse('appdrugres:index') )

def handle_uploaded_file_ct(odir, f):
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fext=alg.getFileExt(f.name)
    fout='%s/%s%s' % (odir, alg.fileNameInputCT, fext)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def handle_uploaded_file_xr(odir, f):
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fext=alg.getFileExt(f.name)
    fout='%s/%s%s' % (odir, alg.fileNameInputXR_Orig, fext)
    print fout
    with open(fout, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def FinishUploadData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect(reverse('appdrugres:index'))
    wdir='%s/%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key, settings.SDRUGRES_TMPDIR)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect(reverse('appdrugres:index'))
    widx=alg.getUniqueDirNameIndex()
    odir='%s/%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key, widx)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    lstCTXR=alg.getDataNamesCTXR(wdir)
    if (lstCTXR[0]!=None) and (lstCTXR[1]!=None):
        shutil.move(lstCTXR[0], odir)
        shutil.move(lstCTXR[1], odir)
        alg.postUplodProcessing(odir)
        taskManagerDrugRes.appendTaskProcessDrugRes(settings.STATIC_ROOT_SEGMXR_DBDATA, odir)
    return HttpResponseRedirect(reverse('appdrugres:index'))

def cleanUplodedData(request):
    if not 'is_logged' in request.session:
        return HttpResponseRedirect( reverse('appdrugres:index') )  ##('/')
    wdir='%s/%s' % (settings.STATIC_ROOT_DRUGRES_USERDATA, request.session.session_key)
    if not os.path.isdir(wdir):
        return HttpResponseRedirect( reverse('appdrugres:index') ) ##('/')
    shutil.rmtree(wdir)
    ret=HttpResponseRedirect( reverse('appdrugres:index') )
    ret.delete_cookie('widx')
    return  ret ##('/')

if __name__=='__main__':
    print alg.fileNameInputCT
    print alg.fileNameInputXR_Orig
