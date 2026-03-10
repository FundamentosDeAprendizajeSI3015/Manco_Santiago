"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { TreePine, Zap, CheckCircle2, XCircle } from "lucide-react"
import { useMLStore } from "@/lib/ml-store"

const modelInfo = {
  random_forest: {
    name: "Random Forest",
    icon: <TreePine className="h-5 w-5" />,
    pros: [
      "Robusto a outliers",
      "No requiere normalizacion",
      "Maneja bien datos faltantes",
      "Interpretable (feature importance)",
    ],
    cons: [
      "Puede ser lento con muchos arboles",
      "Menos preciso en datasets desbalanceados",
    ],
  },
  gradient_boosting: {
    name: "Gradient Boosting",
    icon: <Zap className="h-5 w-5" />,
    pros: [
      "Mejor accuracy en general",
      "Excelente para datos tabulares",
      "Optimizacion secuencial",
      "Mayor capacidad predictiva",
    ],
    cons: [
      "Mas propenso a overfitting",
      "Requiere mas tunning",
      "Entrenamiento mas lento",
    ],
  },
}

export function ModelComparison() {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <div className="grid gap-6 lg:grid-cols-2">
        {(["random_forest", "gradient_boosting"] as const).map((key) => {
          const info = modelInfo[key]
          return (
            <Card key={key} className="border-border bg-card">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "rounded-lg p-2",
                    key === "random_forest" ? "bg-primary/20 text-primary" : "bg-accent/20 text-accent"
                  )}>
                    {info.icon}
                  </div>
                  <div>
                    <CardTitle className="text-foreground">{info.name}</CardTitle>
                    <CardDescription>Clasificador de Ensemble</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex h-[200px] items-center justify-center text-muted-foreground">
                  Entrena un modelo para ver las metricas
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    )
  }

  const rfMetrics = training.random_forest.test
  const gbMetrics = training.gradient_boosting.test
  const bestModelKey = gbMetrics.auc >= rfMetrics.auc ? "gradient_boosting" : "random_forest"

  const models = [
    {
      key: "random_forest" as const,
      metrics: rfMetrics,
      bestParams: training.best_params.random_forest,
    },
    {
      key: "gradient_boosting" as const,
      metrics: gbMetrics,
      bestParams: training.best_params.gradient_boosting,
    },
  ]

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {models.map(({ key, metrics, bestParams }) => {
        const info = modelInfo[key]
        const isBest = key === bestModelKey

        return (
          <Card
            key={key}
            className={cn(
              "border-border bg-card transition-all",
              isBest && "ring-2 ring-primary/50"
            )}
          >
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "rounded-lg p-2",
                    key === "random_forest" ? "bg-primary/20 text-primary" : "bg-accent/20 text-accent"
                  )}>
                    {info.icon}
                  </div>
                  <div>
                    <CardTitle className="text-foreground">{info.name}</CardTitle>
                    <CardDescription>Clasificador de Ensemble</CardDescription>
                  </div>
                </div>
                {isBest && (
                  <Badge className="bg-primary/20 text-primary">
                    Mejor Modelo
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Metrics Grid */}
              <div className="grid grid-cols-5 gap-2">
                {[
                  { label: "Accuracy", value: metrics.accuracy },
                  { label: "Precision", value: metrics.precision },
                  { label: "Recall", value: metrics.recall },
                  { label: "F1 Score", value: metrics.f1 },
                  { label: "AUC", value: metrics.auc },
                ].map((metric) => (
                  <div key={metric.label} className="text-center">
                    <div className="text-xl font-bold text-foreground">
                      {(metric.value * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">{metric.label}</div>
                  </div>
                ))}
              </div>

              {/* Best Parameters */}
              <div>
                <h4 className="mb-2 text-sm font-medium text-foreground">
                  Mejores Hiperparametros
                </h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(bestParams).map(([paramKey, value]) => (
                    <div
                      key={paramKey}
                      className="rounded-md bg-secondary px-2 py-1 text-xs"
                    >
                      <span className="text-muted-foreground">{paramKey}:</span>{" "}
                      <span className="font-medium text-foreground">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Pros & Cons */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <h4 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-accent">
                    <CheckCircle2 className="h-4 w-4" />
                    Ventajas
                  </h4>
                  <ul className="space-y-1">
                    {info.pros.map((pro) => (
                      <li key={pro} className="text-xs text-muted-foreground">
                        {pro}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h4 className="mb-2 flex items-center gap-1.5 text-sm font-medium text-destructive">
                    <XCircle className="h-4 w-4" />
                    Desventajas
                  </h4>
                  <ul className="space-y-1">
                    {info.cons.map((con) => (
                      <li key={con} className="text-xs text-muted-foreground">
                        {con}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
