"""
URL routes for ML API
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DatasetViewSet, TrainingRunViewSet, PredictionViewSet,
    health_check, dashboard_stats
)

router = DefaultRouter()
router.register(r'datasets', DatasetViewSet)
router.register(r'training-runs', TrainingRunViewSet)
router.register(r'predictions', PredictionViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('health/', health_check, name='health-check'),
    path('stats/', dashboard_stats, name='dashboard-stats'),
]
