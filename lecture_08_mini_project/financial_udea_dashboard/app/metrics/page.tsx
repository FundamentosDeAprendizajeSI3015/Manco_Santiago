"use client"

import { Sidebar } from "@/components/dashboard/sidebar"
import { Header } from "@/components/dashboard/header"
import { MetricCard } from "@/components/dashboard/metric-card"
import {
  ROCCurveChart,
  MetricsComparisonChart,
  ConfusionMatrixChart,
  TrainingHistoryChart,
} from "@/components/dashboard/charts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Target, Activity, Gauge, AlertCircle } from "lucide-react"
import { useMLStore, type ModelResult } from "@/lib/ml-store"
import Link from "next/link"
import { Button } from "@/components/ui/button"

function MetricsTable({ data, title }: { data: ModelResult; title: string }) {
  const sets = ["train", "validation", "test"] as const

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">{title}</CardTitle>
        <CardDescription>Metricas por conjunto de datos</CardDescription>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="pb-3 text-left font-medium text-muted-foreground">Set</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">Accuracy</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">Precision</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">Recall</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">F1</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">AUC</th>
            </tr>
          </thead>
          <tbody>
            {sets.map((set) => {
              const metrics = data[set]
              return (
                <tr key={set} className="border-b border-border/50">
                  <td className="py-3 font-medium capitalize text-foreground">{set}</td>
                  <td className="py-3 text-right text-foreground">
                    {(metrics.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 text-right text-foreground">
                    {(metrics.precision * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 text-right text-foreground">
                    {(metrics.recall * 100).toFixed(1)}%
                  </td>
                  <td className="py-3 text-right text-foreground">
                    {metrics.f1.toFixed(3)}
                  </td>
                  <td className="py-3 text-right text-foreground">
                    {metrics.auc.toFixed(3)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}

function ClassificationReport({ model, modelName }: { model: ModelResult; modelName: string }) {
  const cm = model.test.confusion_matrix
  const tn = cm[0]?.[0] || 0
  const fp = cm[0]?.[1] || 0
  const fn = cm[1]?.[0] || 0
  const tp = cm[1]?.[1] || 0

  const precision0 = tn / (tn + fn) || 0
  const recall0 = tn / (tn + fp) || 0
  const f1_0 = 2 * precision0 * recall0 / (precision0 + recall0) || 0

  const precision1 = tp / (tp + fp) || 0
  const recall1 = tp / (tp + fn) || 0
  const f1_1 = 2 * precision1 * recall1 / (precision1 + recall1) || 0

  const support0 = tn + fp
  const support1 = tp + fn

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Reporte de Clasificacion</CardTitle>
        <CardDescription>{modelName} - Test Set</CardDescription>
      </CardHeader>
      <CardContent>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="pb-3 text-left font-medium text-muted-foreground">Clase</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">Precision</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">Recall</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">F1-Score</th>
              <th className="pb-3 text-right font-medium text-muted-foreground">Support</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border/50">
              <td className="py-3 text-foreground">Estable (0)</td>
              <td className="py-3 text-right text-foreground">{precision0.toFixed(2)}</td>
              <td className="py-3 text-right text-foreground">{recall0.toFixed(2)}</td>
              <td className="py-3 text-right text-foreground">{f1_0.toFixed(2)}</td>
              <td className="py-3 text-right text-muted-foreground">{support0}</td>
            </tr>
            <tr className="border-b border-border/50">
              <td className="py-3 text-foreground">Critico (1)</td>
              <td className="py-3 text-right text-foreground">{precision1.toFixed(2)}</td>
              <td className="py-3 text-right text-foreground">{recall1.toFixed(2)}</td>
              <td className="py-3 text-right text-foreground">{f1_1.toFixed(2)}</td>
              <td className="py-3 text-right text-muted-foreground">{support1}</td>
            </tr>
            <tr className="font-medium">
              <td className="py-3 text-foreground">Promedio</td>
              <td className="py-3 text-right text-foreground">{((precision0 + precision1) / 2).toFixed(2)}</td>
              <td className="py-3 text-right text-foreground">{((recall0 + recall1) / 2).toFixed(2)}</td>
              <td className="py-3 text-right text-foreground">{((f1_0 + f1_1) / 2).toFixed(2)}</td>
              <td className="py-3 text-right text-muted-foreground">{support0 + support1}</td>
            </tr>
          </tbody>
        </table>
      </CardContent>
    </Card>
  )
}

export default function MetricsPage() {
  const { training, trainingComplete, isTraining, datasetUploaded } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <div className="min-h-screen bg-background">
        <Sidebar />
        <main className="pl-64">
          <Header
            title="Metricas de Evaluacion"
            description="Analisis detallado del rendimiento de los modelos"
          />
          <div className="p-6">
            <Card className="border-primary/50 bg-primary/5">
              <CardContent className="flex flex-col items-center justify-center p-12 text-center">
                <AlertCircle className="mb-4 h-12 w-12 text-primary" />
                <h3 className="mb-2 text-lg font-semibold text-foreground">
                  {isTraining ? "Entrenando modelos..." : "No hay datos de entrenamiento"}
                </h3>
                <p className="mb-6 max-w-md text-muted-foreground">
                  {isTraining
                    ? "Por favor espera mientras se entrenan los modelos."
                    : "Sube un dataset CSV y entrena los modelos para ver las metricas detalladas."}
                </p>
                {!isTraining && (
                  <Link href="/upload">
                    <Button>{datasetUploaded ? "Ir a Entrenar" : "Subir Dataset"}</Button>
                  </Link>
                )}
              </CardContent>
            </Card>
          </div>
        </main>
      </div>
    )
  }

  const rfTest = training.random_forest.test
  const gbTest = training.gradient_boosting.test

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-64">
        <Header
          title="Metricas de Evaluacion"
          description="Analisis detallado del rendimiento de los modelos"
        />
        <div className="p-6 space-y-6">
          {/* Summary Cards */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              title="Test Accuracy (RF)"
              value={`${(rfTest.accuracy * 100).toFixed(1)}%`}
              description="Random Forest"
              icon={Target}
              variant="default"
            />
            <MetricCard
              title="Test Accuracy (GB)"
              value={`${(gbTest.accuracy * 100).toFixed(1)}%`}
              description="Gradient Boosting"
              icon={Target}
              variant="primary"
            />
            <MetricCard
              title="Mejor F1 Score"
              value={Math.max(rfTest.f1, gbTest.f1).toFixed(3)}
              description={rfTest.f1 >= gbTest.f1 ? "Random Forest" : "Gradient Boosting"}
              icon={Activity}
              variant="accent"
            />
            <MetricCard
              title="Mejor AUC-ROC"
              value={Math.max(rfTest.auc, gbTest.auc).toFixed(3)}
              description={rfTest.auc >= gbTest.auc ? "Random Forest" : "Gradient Boosting"}
              icon={Gauge}
              variant="accent"
            />
          </div>

          {/* Tabs for different views */}
          <Tabs defaultValue="comparison" className="space-y-6">
            <TabsList className="bg-secondary">
              <TabsTrigger value="comparison">Comparacion</TabsTrigger>
              <TabsTrigger value="rf">Random Forest</TabsTrigger>
              <TabsTrigger value="gb">Gradient Boosting</TabsTrigger>
              <TabsTrigger value="history">Historial</TabsTrigger>
            </TabsList>

            <TabsContent value="comparison" className="space-y-6">
              <div className="grid gap-6 lg:grid-cols-2">
                <MetricsComparisonChart />
                <ROCCurveChart />
              </div>
            </TabsContent>

            <TabsContent value="rf" className="space-y-6">
              <MetricsTable
                data={training.random_forest}
                title="Random Forest - Metricas Detalladas"
              />
              <div className="grid gap-6 lg:grid-cols-2">
                <ConfusionMatrixChart modelType="random_forest" />
                <ClassificationReport model={training.random_forest} modelName="Random Forest" />
              </div>
            </TabsContent>

            <TabsContent value="gb" className="space-y-6">
              <MetricsTable
                data={training.gradient_boosting}
                title="Gradient Boosting - Metricas Detalladas"
              />
              <div className="grid gap-6 lg:grid-cols-2">
                <ConfusionMatrixChart modelType="gradient_boosting" />
                <ClassificationReport model={training.gradient_boosting} modelName="Gradient Boosting" />
              </div>
            </TabsContent>

            <TabsContent value="history" className="space-y-6">
              <TrainingHistoryChart />
            </TabsContent>
          </Tabs>
        </div>
      </main>
    </div>
  )
}
