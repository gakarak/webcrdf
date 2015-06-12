# from bzrlib.osutils import isfile
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.shortcuts import render_to_response
from django.core.urlresolvers import reverse

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
