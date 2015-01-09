from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from webcrdf import views

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'webcrdf.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

##    url(r'^admin/', 	include(admin.site.urls)),
    url(r'^$',		views.Index,  name='main-home'),
    url(r'^login/',	views.Login,  name='main-login'),
    url(r'^logout/',	views.Logout, name='main-logout'),
    url(r'^appwebcam/',	views.WebCamSearch, name='main-webcam'),
    url(r'^appcbir/',	include('appcbir.urls',  namespace='appcbir', app_name='appcbir')),
    url(r'^appxray/',	include('appxray.urls',  namespace='appxray', app_name='appxray')),
    url(r'^appsegmxr/',	include('appsegmxr.urls',  namespace='appsegmxr', app_name='appsegmxr')),
    url(r'^appmelanoma/',	include('appmelanoma.urls',  namespace='appmelanoma', app_name='appmelanoma')),
)

