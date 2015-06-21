# from bzrlib.osutils import isfile
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings


import os
import json
from webcrdf.settings import VIDEOCBIR
import webcrdf.settings as sett

# Create your views here.
################################################
def Index(request):
    # print '(appvideocbir) static URL: %s' % settings.STATIC_ROOT_VIDEOCBIR_USERDATA
    # print 'path: [%s]' % request.path
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        context={'userName': userName, 'uploadedImages': None}
        return render(request, 'appvideocbir.html', context)
    else:
        return HttpResponseRedirect('/login/?next=%s' % request.path)

def apiSearch(request):
    print request.POST, " : ", len(request.POST)
    if not 'is_logged' in request.session:
        return HttpResponse(json.dumps(()))
    if request.method=='POST':
        qIdxVideo=int(request.POST['idVideo'])
        qIdxFrame=int(request.POST['idFrame'])
        # retJSON=HISTOLOGY.processSelection(qIdxSlide, qIdxRow, qIdxCol, odir)
        retJSON=None
        return HttpResponse(json.dumps(retJSON))
    else:
        return HttpResponse(json.dumps(()))