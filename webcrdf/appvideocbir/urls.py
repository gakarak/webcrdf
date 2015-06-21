from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from appvideocbir import views

urlpatterns = patterns('',
    url(r'^$',           views.Index,           name='index'),
    url(r'^apisearch/',  views.apiSearch,       name='apiSearch'),
)
