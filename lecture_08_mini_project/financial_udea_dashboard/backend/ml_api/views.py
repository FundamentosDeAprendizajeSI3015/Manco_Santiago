"""
API Views for FIRE_UdeA ML Dashboard
"""
from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.utils import timezone
from django.conf import settings
import os

from .models import Dataset, TrainingRun, Prediction
from .serializers import (
    DatasetSerializer, DatasetUploadSerializer, TrainingRunSerializer,
    PredictionSerializer, PredictionInputSerializer, EDAResultSerializer
)
from .ml_service import ml_service


class DatasetViewSet(viewsets.ModelViewSet):
    """ViewSet for Dataset operations"""
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    parser_classes = [MultiPartParser, FormParser]
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """Upload a new dataset"""
        serializer = DatasetUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        file = serializer.validated_data['file']
        name = serializer.validated_data.get('name', file.name)
        
        # Create dataset instance
        dataset = Dataset.objects.create(
            name=name,
            file=file
        )
        
        # Process file to get row and column count
        try:
            import pandas as pd
            df = pd.read_csv(dataset.file.path)
            dataset.rows_count = len(df)
            dataset.columns_count = len(df.columns)
            dataset.processed = True
            dataset.save()
        except Exception as e:
            dataset.delete()
            return Response(
                {"error": f"Error procesando archivo: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        return Response(DatasetSerializer(dataset).data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['get'])
    def eda(self, request, pk=None):
        """Get EDA results for a dataset"""
        dataset = self.get_object()
        
        try:
            eda_results = ml_service.compute_eda(dataset.file.path)
            return Response(eda_results)
        except Exception as e:
            return Response(
                {"error": f"Error en EDA: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'])
    def preview(self, request, pk=None):
        """Get preview of dataset (first 10 rows)"""
        dataset = self.get_object()
        
        try:
            import pandas as pd
            df = pd.read_csv(dataset.file.path)
            preview_data = df.head(10).to_dict(orient='records')
            return Response({
                "columns": df.columns.tolist(),
                "data": preview_data,
                "total_rows": len(df)
            })
        except Exception as e:
            return Response(
                {"error": f"Error leyendo archivo: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TrainingRunViewSet(viewsets.ModelViewSet):
    """ViewSet for Training Run operations"""
    queryset = TrainingRun.objects.all()
    serializer_class = TrainingRunSerializer
    
    @action(detail=False, methods=['post'])
    def train(self, request):
        """Start training for both models"""
        dataset_id = request.data.get('dataset_id')
        
        if not dataset_id:
            return Response(
                {"error": "dataset_id es requerido"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            dataset = Dataset.objects.get(pk=dataset_id)
        except Dataset.DoesNotExist:
            return Response(
                {"error": "Dataset no encontrado"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        results = {}
        
        try:
            # Load and preprocess data
            data, X, y, num_features, cat_features = ml_service.load_and_preprocess_data(
                dataset.file.path
            )
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = ml_service.split_data(X, y)
            
            # Create preprocessor
            preprocessor = ml_service.create_preprocessor(num_features, cat_features)
            
            # Train Random Forest
            rf_run = TrainingRun.objects.create(
                dataset=dataset,
                model_type='random_forest',
                status='running'
            )
            
            try:
                rf_model, rf_params = ml_service.train_random_forest(
                    X_train, y_train, preprocessor
                )
                
                # Evaluate on all sets
                rf_train_metrics = ml_service.evaluate_model(rf_model, X_train, y_train, "train")
                rf_val_metrics = ml_service.evaluate_model(rf_model, X_val, y_val, "val")
                rf_test_metrics = ml_service.evaluate_model(rf_model, X_test, y_test, "test")
                
                # Get feature importance
                rf_importance = ml_service.get_feature_importance(rf_model)
                
                # Save model
                rf_model_path = ml_service.save_model(rf_model, "random_forest", str(dataset.id))
                
                # Update training run
                rf_run.accuracy_train = rf_train_metrics["accuracy"]
                rf_run.accuracy_val = rf_val_metrics["accuracy"]
                rf_run.accuracy_test = rf_test_metrics["accuracy"]
                rf_run.precision = rf_test_metrics["precision"]
                rf_run.recall = rf_test_metrics["recall"]
                rf_run.f1_score = rf_test_metrics["f1_score"]
                rf_run.auc_roc = rf_test_metrics["auc_roc"]
                rf_run.best_params = rf_params
                rf_run.feature_importance = rf_importance
                rf_run.model_path = rf_model_path
                rf_run.status = 'completed'
                rf_run.completed_at = timezone.now()
                rf_run.save()
                
                results['random_forest'] = TrainingRunSerializer(rf_run).data
                
            except Exception as e:
                rf_run.status = 'failed'
                rf_run.save()
                results['random_forest_error'] = str(e)
            
            # Train Gradient Boosting
            gb_run = TrainingRun.objects.create(
                dataset=dataset,
                model_type='gradient_boosting',
                status='running'
            )
            
            try:
                gb_model, gb_params = ml_service.train_gradient_boosting(
                    X_train, y_train, preprocessor
                )
                
                # Evaluate on all sets
                gb_train_metrics = ml_service.evaluate_model(gb_model, X_train, y_train, "train")
                gb_val_metrics = ml_service.evaluate_model(gb_model, X_val, y_val, "val")
                gb_test_metrics = ml_service.evaluate_model(gb_model, X_test, y_test, "test")
                
                # Get feature importance
                gb_importance = ml_service.get_feature_importance(gb_model)
                
                # Save model
                gb_model_path = ml_service.save_model(gb_model, "gradient_boosting", str(dataset.id))
                
                # Update training run
                gb_run.accuracy_train = gb_train_metrics["accuracy"]
                gb_run.accuracy_val = gb_val_metrics["accuracy"]
                gb_run.accuracy_test = gb_test_metrics["accuracy"]
                gb_run.precision = gb_test_metrics["precision"]
                gb_run.recall = gb_test_metrics["recall"]
                gb_run.f1_score = gb_test_metrics["f1_score"]
                gb_run.auc_roc = gb_test_metrics["auc_roc"]
                gb_run.best_params = gb_params
                gb_run.feature_importance = gb_importance
                gb_run.model_path = gb_model_path
                gb_run.status = 'completed'
                gb_run.completed_at = timezone.now()
                gb_run.save()
                
                results['gradient_boosting'] = TrainingRunSerializer(gb_run).data
                
            except Exception as e:
                gb_run.status = 'failed'
                gb_run.save()
                results['gradient_boosting_error'] = str(e)
            
            return Response(results, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            return Response(
                {"error": f"Error en entrenamiento: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def comparison(self, request):
        """Get comparison of latest models"""
        rf_run = TrainingRun.objects.filter(
            model_type='random_forest',
            status='completed'
        ).first()
        
        gb_run = TrainingRun.objects.filter(
            model_type='gradient_boosting',
            status='completed'
        ).first()
        
        return Response({
            "random_forest": TrainingRunSerializer(rf_run).data if rf_run else None,
            "gradient_boosting": TrainingRunSerializer(gb_run).data if gb_run else None
        })


class PredictionViewSet(viewsets.ModelViewSet):
    """ViewSet for Prediction operations"""
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """Make predictions with both models"""
        input_serializer = PredictionInputSerializer(data=request.data)
        input_serializer.is_valid(raise_exception=True)
        input_data = input_serializer.validated_data
        
        results = {}
        
        # Get latest completed models
        rf_run = TrainingRun.objects.filter(
            model_type='random_forest',
            status='completed'
        ).first()
        
        gb_run = TrainingRun.objects.filter(
            model_type='gradient_boosting',
            status='completed'
        ).first()
        
        # Predict with Random Forest
        if rf_run and rf_run.model_path and os.path.exists(rf_run.model_path):
            try:
                rf_model = ml_service.load_model(rf_run.model_path)
                rf_pred, rf_prob = ml_service.predict(rf_model, input_data)
                
                # Save prediction
                rf_prediction = Prediction.objects.create(
                    training_run=rf_run,
                    input_data=input_data,
                    prediction=rf_pred,
                    probability=rf_prob
                )
                
                results['random_forest'] = {
                    "prediction": rf_pred,
                    "probability": rf_prob,
                    "label": "Estable" if rf_pred == 0 else "Critico"
                }
            except Exception as e:
                results['random_forest_error'] = str(e)
        else:
            results['random_forest'] = None
        
        # Predict with Gradient Boosting
        if gb_run and gb_run.model_path and os.path.exists(gb_run.model_path):
            try:
                gb_model = ml_service.load_model(gb_run.model_path)
                gb_pred, gb_prob = ml_service.predict(gb_model, input_data)
                
                # Save prediction
                gb_prediction = Prediction.objects.create(
                    training_run=gb_run,
                    input_data=input_data,
                    prediction=gb_pred,
                    probability=gb_prob
                )
                
                results['gradient_boosting'] = {
                    "prediction": gb_pred,
                    "probability": gb_prob,
                    "label": "Estable" if gb_pred == 0 else "Critico"
                }
            except Exception as e:
                results['gradient_boosting_error'] = str(e)
        else:
            results['gradient_boosting'] = None
        
        return Response(results)


@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    return Response({
        "status": "healthy",
        "message": "FIRE_UdeA ML API is running"
    })


@api_view(['GET'])
def dashboard_stats(request):
    """Get dashboard statistics"""
    datasets_count = Dataset.objects.count()
    training_runs_count = TrainingRun.objects.filter(status='completed').count()
    predictions_count = Prediction.objects.count()
    
    # Get best models
    rf_best = TrainingRun.objects.filter(
        model_type='random_forest',
        status='completed'
    ).order_by('-accuracy_test').first()
    
    gb_best = TrainingRun.objects.filter(
        model_type='gradient_boosting',
        status='completed'
    ).order_by('-accuracy_test').first()
    
    return Response({
        "datasets_count": datasets_count,
        "training_runs_count": training_runs_count,
        "predictions_count": predictions_count,
        "best_random_forest": TrainingRunSerializer(rf_best).data if rf_best else None,
        "best_gradient_boosting": TrainingRunSerializer(gb_best).data if gb_best else None
    })
