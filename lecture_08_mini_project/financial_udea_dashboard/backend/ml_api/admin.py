from django.contrib import admin
from .models import Dataset, TrainingRun, Prediction


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'rows_count', 'columns_count', 'processed', 'uploaded_at']
    list_filter = ['processed', 'uploaded_at']
    search_fields = ['name']
    readonly_fields = ['id', 'uploaded_at']


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ['model_type', 'dataset', 'status', 'accuracy_test', 'auc_roc', 'started_at']
    list_filter = ['model_type', 'status', 'started_at']
    search_fields = ['dataset__name']
    readonly_fields = ['id', 'started_at', 'completed_at']


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'training_run', 'prediction', 'probability', 'created_at']
    list_filter = ['prediction', 'created_at']
    readonly_fields = ['id', 'created_at']
