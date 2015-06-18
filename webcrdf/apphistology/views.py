# from bzrlib.osutils import isfile
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse


import os
import json
from webcrdf.settings import HISTOLOGY
import webcrdf.settings as sett

# Create your views here.
################################################
def Index(request):
    print '(apphistology) static URL: %s' % settings.STATIC_ROOT_SEGMCT_USERDATA
    print 'path: [%s]' % request.path
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        context={'userName': userName, 'uploadedImages': None}
        return render(request, 'apphistology.html', context)
    else:
        # context={'next': '/apphistology/'}
        # return render(request, 'login.html', context)
        return HttpResponseRedirect('/login/?next=%s' % request.path)

def apiSearch(request):
    print request.POST, " : ", len(request.POST)
    if not 'is_logged' in request.session:
        return HttpResponse(json.dumps(()))
    if request.method=='POST':
        qIdxSlide=int(request.POST['idSlide'])
        qIdxRow=int(request.POST['idRow'])
        qIdxCol=int(request.POST['idCol'])
        if not os.path.isdir(sett.STATIC_ROOT_HISTOLOGY_USERDATA):
            os.mkdir(sett.STATIC_ROOT_HISTOLOGY_USERDATA)
        odir="%s/%s" % (sett.STATIC_ROOT_HISTOLOGY_USERDATA, request.session.session_key)
        print request.POST
        retJSON=HISTOLOGY.processSelection(qIdxSlide, qIdxRow, qIdxCol, odir)
        return HttpResponse(json.dumps(retJSON))
    else:
        return HttpResponse(json.dumps(()))