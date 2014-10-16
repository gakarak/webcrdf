from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

##from test_sessions import views
from appsegmxr import views

urlpatterns = patterns('',
    url(r'^$',          views.Index,			    name='index'),
    url(r'^upload/',    views.Upload,			    name='upload'),
    url(r'^uploadfdb/', views.UploadFromDB,		    name='uploadfdb'),
    url(r'^clean/',     views.cleanUplodedData,     name='clean'),
    url(r'^getinfo/',   views.getInfoAboutImages,   name='getinfo'),
    url(r'^gallery/',   views.ImageGallery,         name='gallery'),
)
