from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

##from test_sessions import views
from appxray import views

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'test_sessions.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

##    url(r'^admin/', include(admin.site.urls)),
    url(r'^$',          views.Index,			name='index'),
    url(r'^upload/',    views.Upload,			name='upload'),
    url(r'^uploadfdb/', views.UploadFromDB,		name='uploadfdb'),
    url(r'^clean/',     views.cleanUplodedData, name='clean'),
    url(r'^roisearch/', views.ROISearch,		name='roisearch'),
    url(r'^gallery/',	views.ImageGallery,		name='gallery'),
)
