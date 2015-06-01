__author__ = 'ar'

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django import forms
from django.shortcuts import render_to_response

from django.conf.urls.static import static
from django.conf import settings


def Index(request):
    ##print 'static URL: %s' % settings.STATIC_ROOT_USER
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        ##uploadedImages=getUploadedImageList(request.session.session_key)
        ##context={'userName': userName, 'uploadedImages': uploadedImages}
        context={'userName': userName}
        ##return render(request, 'index.html')
        return render(request, 'index.html', context)
    else:
        # return render(request, 'login.html')
        return HttpResponseRedirect('/login/')

def Login(request):
    print '::Login: session-Id = (%s)' % request.session.session_key
    if not 'is_logged' in request.session:
        request.session['is_logged'] = '1'
        request.session['username'] = 'Anonymous'
    print "next=[%s]" % request.GET.get('next')
    print "next-path=[%s]" % request.path
    if request.GET:
        print request.GET
        tmpRediretUrl=request.GET.get('next')
        if tmpRediretUrl is not None:
            return HttpResponseRedirect(tmpRediretUrl)
    return HttpResponseRedirect('/')

def Logout(request):
    print '::Logout: session-Id = (%s)' % request.session.session_key
    if 'is_logged' in request.session:
        del request.session['is_logged']
    if 'username' in request.session:
        del request.session['username']
    return HttpResponseRedirect('/')

def WebCamSearch(request):
    if 'is_logged' in request.session:
        userName = 'Unknown'
        if 'username' in request.session:
            userName = request.session['username']
        context={'userName': userName}
        return render(request, 'appwebcam2.html', context)
    else:
        return render(request, 'login.html')
