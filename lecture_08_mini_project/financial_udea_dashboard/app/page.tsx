"use client"

import { Sidebar } from "@/components/dashboard/sidebar"
import { Header } from "@/components/dashboard/header"
import { MetricCard } from "@/components/dashboard/metric-card"
import {
  ROCCurveChart,
  FeatureImportanceChart,
  ConfusionMatrixChart,
  TargetDistributionChart,
  ModelScoresRadial,
  TrainingHistoryChart,
  DataSplitChart,
} from "@/components/dashboard/charts"
import { ModelComparison } from "@/components/dashboard/model-comparison"
import { Activity, Target, Percent, TrendingUp, Database, BarChart3, AlertCircle } from "lucide-react"
import { useMLStore } from "@/lib/ml-store"
import { Card, CardContent } from "@/components/ui/card"
import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function DashboardPage() {
  const { trainingComplete, training, eda, isTraining, datasetUploaded } = useMLStore()

  const getBestAccuracy = () => {
    if (!training) return { value: "--", model: "Sin datos" }
    const rfAcc = training.random_forest.test.accuracy
    const gbAcc = training.gradient_boosting.test.accuracy
    if (gbAcc >= rfAcc) {
      return { value: `${(gbAcc * 100).toFixed(1)}%`, model: "Gradient Boosting" }
    }
    return { value: `${(rfAcc * 100).toFixed(1)}%`, model: "Random Forest" }
  }

  const getBestAUC = () => {
    if (!training) return { value: "--", model: "Sin datos" }
    const rfAuc = training.random_forest.test.auc
    const gbAuc = training.gradient_boosting.test.auc
    if (gbAuc >= rfAuc) {
      return { value: gbAuc.toFixed(3), model: "Gradient Boosting" }
    }
    return { value: rfAuc.toFixed(3), model: "Random Forest" }
  }

  const getAvgPrecision = () => {
    if (!training) return "--"
    const avg = (training.random_forest.test.precision + training.gradient_boosting.test.precision) / 2
    return `${(avg * 100).toFixed(1)}%`
  }

  const getAvgF1 = () => {
    if (!training) return "--"
    const avg = (training.random_forest.test.f1 + training.gradient_boosting.test.f1) / 2
    return avg.toFixed(3)
  }

  const getTotalRecords = () => {
    if (!eda) return "--"
    return eda.total_samples.toLocaleString()
  }

  const getFeaturesCount = () => {
    if (!eda) return "--"
    return eda.features_count.toString()
  }

  const bestAcc = getBestAccuracy()
  const bestAuc = getBestAUC()

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-64">
        <Header
          title="Dashboard de Machine Learning"
          description="FIRE_UdeA - Clasificacion Financiera"
        />
        <div className="p-6 space-y-6">
          {!trainingComplete && (
            <Card className="border-primary/50 bg-primary/5">
              <CardContent className="flex items-center justify-between p-4">
                <div className="flex items-center gap-3">
                  <AlertCircle className="h-5 w-5 text-primary" />
                  <div>
                    <p className="font-medium text-foreground">
                      {isTraining 
                        ? "Entrenando modelos..." 
                        : datasetUploaded 
                          ? "Dataset cargado. Inicia el entrenamiento para ver los resultados."
                          : "No hay datos de entrenamiento"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {isTraining
                        ? "Por favor espera mientras se entrenan los modelos Random Forest y Gradient Boosting."
                        : "Sube un dataset CSV y entrena los modelos para ver las metricas y graficos."}
                    </p>
                  </div>
                </div>
                {!isTraining && (
                  <Link href="/upload">
                    <Button>
                      {datasetUploaded ? "Ir a Entrenar" : "Subir Dataset"}
                    </Button>
                  </Link>
                )}
              </CardContent>
            </Card>
          )}

          {/* KPI Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-6">
            <MetricCard
              title="Mejor Accuracy"
              value={bestAcc.value}
              description={bestAcc.model}
              icon={Target}
              variant={trainingComplete ? "primary" : "default"}
            />
            <MetricCard
              title="Mejor AUC"
              value={bestAuc.value}
              description={bestAuc.model}
              icon={TrendingUp}
              variant={trainingComplete ? "accent" : "default"}
            />
            <MetricCard
              title="Precision"
              value={getAvgPrecision()}
              description="Promedio de modelos"
              icon={Percent}
              variant="default"
            />
            <MetricCard
              title="F1 Score"
              value={getAvgF1()}
              description="Media ponderada"
              icon={Activity}
              variant="default"
            />
            <MetricCard
              title="Total Registros"
              value={getTotalRecords()}
              description="60/20/20 split"
              icon={Database}
              variant="default"
            />
            <MetricCard
              title="Features"
              value={getFeaturesCount()}
              description="Variables numericas"
              icon={BarChart3}
              variant="default"
            />
          </div>

          {/* Model Comparison */}
          <section>
            <h2 className="mb-4 text-lg font-semibold text-foreground">
              Comparacion de Modelos
            </h2>
            <ModelComparison />
          </section>

          {/* Charts Row 1 */}
          <div className="grid gap-6 lg:grid-cols-2">
            <ROCCurveChart />
            <FeatureImportanceChart />
          </div>

          {/* Charts Row 2 */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <ConfusionMatrixChart />
            <TargetDistributionChart />
            <ModelScoresRadial />
          </div>

          {/* Training History & Data Split */}
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <TrainingHistoryChart />
            </div>
            <DataSplitChart />
          </div>
        </div>
      </main>
    </div>
  )
}
