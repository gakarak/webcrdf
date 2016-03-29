from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

##from test_sessions import views
from appdrugres import views

urlpatterns = patterns('',
    url(r'^$',              views.Index,			    name='index'),
    url(r'^uploadct/',      views.UploadCT,		        name='uploadct'),
    url(r'^uploadxr/',      views.UploadXR,		        name='uploadxr'),
    url(r'^finishupload/',  views.FinishUploadData,	    name='finishuploaddata'),
    url(r'^uploadfdb/',     views.UploadFromDB,		    name='uploadfdb'),
    url(r'^clean/',         views.cleanUplodedData,		name='clean'),
    url(r'^getinfo/',       views.getInfoAboutImages,	name='getinfo'),
    url(r'^gallery/',       views.ImageGallery,		    name='gallery'),
    url(r'^showct/',        views.ShowCT,		        name='showct'),
)
