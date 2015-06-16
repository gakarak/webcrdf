from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from apphistology import views

urlpatterns = patterns('',
    url(r'^$',           views.Index,           name='index'),
    url(r'^apisearch/',  views.apiSearch,       name='apiSearch'),
    # url(r'^getinfo/',   views.getInfoAboutImages,   name='getinfo'),
    # url(r'^gallery/',   views.ImageGallery,         name='gallery'),
)
