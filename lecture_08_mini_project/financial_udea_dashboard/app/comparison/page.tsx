"use client"

import { Sidebar } from "@/components/dashboard/sidebar"
import { Header } from "@/components/dashboard/header"
import { ModelComparison } from "@/components/dashboard/model-comparison"
import { MetricsComparisonChart, FeatureImportanceChart, ROCCurveChart } from "@/components/dashboard/charts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TreePine, Zap, Award, AlertCircle } from "lucide-react"
import { useMLStore } from "@/lib/ml-store"
import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function ComparisonPage() {
  const { training, trainingComplete, isTraining, datasetUploaded } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <div className="min-h-screen bg-background">
        <Sidebar />
        <main className="pl-64">
          <Header
            title="Comparacion de Modelos"
            description="Random Forest vs Gradient Boosting"
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
                    : "Sube un dataset CSV y entrena los modelos para ver la comparacion."}
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

  const performanceComparison = [
    {
      metric: "Accuracy",
      rf: `${(rfTest.accuracy * 100).toFixed(1)}%`,
      gb: `${(gbTest.accuracy * 100).toFixed(1)}%`,
      winner: gbTest.accuracy >= rfTest.accuracy ? "gb" : "rf",
      icon: Award,
    },
    {
      metric: "Precision",
      rf: `${(rfTest.precision * 100).toFixed(1)}%`,
      gb: `${(gbTest.precision * 100).toFixed(1)}%`,
      winner: gbTest.precision >= rfTest.precision ? "gb" : "rf",
      icon: Award,
    },
    {
      metric: "Recall",
      rf: `${(rfTest.recall * 100).toFixed(1)}%`,
      gb: `${(gbTest.recall * 100).toFixed(1)}%`,
      winner: gbTest.recall >= rfTest.recall ? "gb" : "rf",
      icon: Award,
    },
    {
      metric: "AUC-ROC",
      rf: rfTest.auc.toFixed(3),
      gb: gbTest.auc.toFixed(3),
      winner: gbTest.auc >= rfTest.auc ? "gb" : "rf",
      icon: Award,
    },
  ]

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-64">
        <Header
          title="Comparacion de Modelos"
          description="Random Forest vs Gradient Boosting"
        />
        <div className="p-6 space-y-6">
          {/* Head to Head */}
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-foreground">Comparacion Directa</CardTitle>
              <CardDescription>
                Rendimiento lado a lado de ambos modelos (Test Set)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                {performanceComparison.map((item) => (
                  <div
                    key={item.metric}
                    className="rounded-lg border border-border bg-secondary/30 p-4"
                  >
                    <div className="mb-3 flex items-center gap-2">
                      <item.icon className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm font-medium text-muted-foreground">
                        {item.metric}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center">
                        <div className="mb-1 flex items-center justify-center gap-1">
                          <TreePine className="h-4 w-4 text-primary" />
                          <span className="text-xs text-muted-foreground">RF</span>
                        </div>
                        <div className="flex items-center justify-center gap-1">
                          <span className="text-lg font-bold text-foreground">{item.rf}</span>
                          {item.winner === "rf" && (
                            <Badge className="h-5 bg-accent/20 text-accent text-xs">
                              Mejor
                            </Badge>
                          )}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="mb-1 flex items-center justify-center gap-1">
                          <Zap className="h-4 w-4 text-accent" />
                          <span className="text-xs text-muted-foreground">GB</span>
                        </div>
                        <div className="flex items-center justify-center gap-1">
                          <span className="text-lg font-bold text-foreground">{item.gb}</span>
                          {item.winner === "gb" && (
                            <Badge className="h-5 bg-accent/20 text-accent text-xs">
                              Mejor
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Model Cards */}
          <ModelComparison />

          {/* Charts */}
          <div className="grid gap-6 lg:grid-cols-2">
            <MetricsComparisonChart />
            <ROCCurveChart />
          </div>

          <FeatureImportanceChart />

          {/* Recommendations */}
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-foreground">Recomendaciones</CardTitle>
              <CardDescription>
                Guia para seleccionar el modelo apropiado
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-lg border border-primary/30 bg-primary/5 p-4">
                  <div className="mb-3 flex items-center gap-2">
                    <div className="rounded-lg bg-primary/20 p-2">
                      <TreePine className="h-5 w-5 text-primary" />
                    </div>
                    <h4 className="font-semibold text-foreground">Usar Random Forest cuando:</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>Necesitas interpretabilidad del modelo</li>
                    <li>El tiempo de entrenamiento es critico</li>
                    <li>Tienes recursos computacionales limitados</li>
                    <li>Prefieres robustez sobre precision maxima</li>
                    <li>Trabajas con datos con muchos outliers</li>
                  </ul>
                </div>
                <div className="rounded-lg border border-accent/30 bg-accent/5 p-4">
                  <div className="mb-3 flex items-center gap-2">
                    <div className="rounded-lg bg-accent/20 p-2">
                      <Zap className="h-5 w-5 text-accent" />
                    </div>
                    <h4 className="font-semibold text-foreground">Usar Gradient Boosting cuando:</h4>
                  </div>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li>La precision maxima es la prioridad</li>
                    <li>Tienes tiempo para optimizar hiperparametros</li>
                    <li>El dataset esta bien balanceado</li>
                    <li>Necesitas el mejor AUC-ROC posible</li>
                    <li>Tienes suficientes recursos computacionales</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
