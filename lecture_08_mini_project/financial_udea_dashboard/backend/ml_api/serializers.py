from rest_framework import serializers
from .models import Dataset, TrainingRun, Prediction


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'
        read_only_fields = ['id', 'uploaded_at', 'rows_count', 'columns_count', 'processed']


class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    name = serializers.CharField(max_length=255, required=False)
    
    def validate_file(self, value):
        if not value.name.endswith('.csv'):
            raise serializers.ValidationError("Solo se permiten archivos CSV")
        return value


class TrainingRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingRun
        fields = '__all__'
        read_only_fields = ['id', 'started_at', 'completed_at', 'status', 
                          'accuracy_train', 'accuracy_val', 'accuracy_test',
                          'precision', 'recall', 'f1_score', 'auc_roc',
                          'best_params', 'feature_importance', 'model_path']


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['id', 'prediction', 'probability', 'created_at']


class PredictionInputSerializer(serializers.Serializer):
    """Serializer for prediction input data"""
    ingresos_totales = serializers.FloatField()
    gastos_personal = serializers.FloatField()
    liquidez = serializers.FloatField()
    dias_efectivo = serializers.IntegerField()
    cfo = serializers.FloatField()
    participacion_ley30 = serializers.FloatField(required=False, default=0.0)
    participacion_regalias = serializers.FloatField(required=False, default=0.0)
    participacion_servicios = serializers.FloatField(required=False, default=0.0)
    participacion_matriculas = serializers.FloatField(required=False, default=0.0)
    hhi_fuentes = serializers.FloatField()
    endeudamiento = serializers.FloatField()
    tendencia_ingresos = serializers.FloatField()
    gp_ratio = serializers.FloatField()


class MetricsComparisonSerializer(serializers.Serializer):
    """Serializer for model metrics comparison"""
    random_forest = serializers.DictField()
    gradient_boosting = serializers.DictField()


class EDAResultSerializer(serializers.Serializer):
    """Serializer for EDA results"""
    statistics = serializers.DictField()
    correlation_matrix = serializers.DictField()
    target_distribution = serializers.DictField()
    missing_values = serializers.DictField()
