from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('', views.upload, name="upload"),
    path('login/', views.login_process, name="loginPage"),
    path('signup/', views.registerPage, name="registerPage"),
    path('logout/', views.logoutuser, name="logout"),
    path('profile/', views.profile, name="profile"),
    path('predictt/', views.predictt, name="predictt"),
    path('rembg/', views.rembg, name="rembg"),
    path('custombg/', views.custombg, name="custombg")
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
