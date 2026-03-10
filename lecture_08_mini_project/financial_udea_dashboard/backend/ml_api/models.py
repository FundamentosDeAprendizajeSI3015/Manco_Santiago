from django.db import models
import uuid


class Dataset(models.Model):
    """Model to store uploaded datasets"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    rows_count = models.IntegerField(default=0)
    columns_count = models.IntegerField(default=0)
    processed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"


class TrainingRun(models.Model):
    """Model to store training runs"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='training_runs')
    model_type = models.CharField(max_length=50, choices=[
        ('random_forest', 'Random Forest'),
        ('gradient_boosting', 'Gradient Boosting'),
    ])
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    # Metrics
    accuracy_train = models.FloatField(null=True, blank=True)
    accuracy_val = models.FloatField(null=True, blank=True)
    accuracy_test = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    auc_roc = models.FloatField(null=True, blank=True)
    
    # Hyperparameters (stored as JSON)
    hyperparameters = models.JSONField(default=dict, blank=True)
    best_params = models.JSONField(default=dict, blank=True)
    
    # Feature importance (stored as JSON)
    feature_importance = models.JSONField(default=dict, blank=True)
    
    # Model file path
    model_path = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-started_at']
    
    def __str__(self):
        return f"{self.model_type} - {self.started_at.strftime('%Y-%m-%d %H:%M')}"


class Prediction(models.Model):
    """Model to store predictions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    training_run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name='predictions')
    input_data = models.JSONField()
    prediction = models.IntegerField()  # 0 or 1
    probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        label = "Estable" if self.prediction == 0 else "Critico"
        return f"{label} ({self.probability:.2%}) - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
