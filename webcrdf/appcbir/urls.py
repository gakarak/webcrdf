from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

from appcbir import views

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'djcbir.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

##    url(r'^admin/',   include(admin.site.urls)),
##    url(r'^login/',   views.Login,  name='Login'),
##    url(r'^logout/',  views.Logout, name='Logout'),
    url(r'^$',              views.Index,            name='index'),
    url(r'^upload/',        views.Upload,           name='upload'),
    url(r'^clean/',         views.cleanUplodedData, name='clean'),
    url(r'^api/dbinfo/',    views.apiRequestDbInfo, name='apiRequestDbInfo'),
    url(r'^api/searchdb/',  views.apiSearchDB,      name='apiSearchDB'),

)
