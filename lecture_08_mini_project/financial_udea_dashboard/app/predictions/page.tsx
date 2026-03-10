"use client"

import { useState } from "react"
import { Sidebar } from "@/components/dashboard/sidebar"
import { Header } from "@/components/dashboard/header"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Play, RotateCcw, AlertTriangle, CheckCircle2, TreePine, Zap } from "lucide-react"

interface PredictionResult {
  randomForest: {
    prediction: number
    probability: number
  }
  gradientBoosting: {
    prediction: number
    probability: number
  }
}

const defaultValues = {
  ingresos_totales: 100000000,
  gastos_personal: 70000000,
  liquidez: 1.2,
  dias_efectivo: 30,
  cfo: 5000000,
  endeudamiento: 0.35,
  gp_ratio: 0.7,
  hhi_fuentes: 0.25,
  tendencia_ingresos: 0.05,
}

export default function PredictionsPage() {
  const [values, setValues] = useState(defaultValues)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handlePredict = async () => {
    setIsLoading(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1500))
    
    // Simulate prediction based on input values
    const riskScore = (
      (values.endeudamiento > 0.5 ? 0.3 : 0) +
      (values.liquidez < 1 ? 0.25 : 0) +
      (values.gp_ratio > 0.9 ? 0.2 : 0) +
      (values.cfo < 0 ? 0.15 : 0) +
      (values.dias_efectivo < 20 ? 0.1 : 0)
    )

    const rfProbability = Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.1 - 0.05)))
    const gbProbability = Math.min(0.95, Math.max(0.05, riskScore + (Math.random() * 0.1 - 0.05)))

    setResult({
      randomForest: {
        prediction: rfProbability > 0.5 ? 1 : 0,
        probability: rfProbability,
      },
      gradientBoosting: {
        prediction: gbProbability > 0.5 ? 1 : 0,
        probability: gbProbability,
      },
    })
    setIsLoading(false)
  }

  const handleReset = () => {
    setValues(defaultValues)
    setResult(null)
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("es-CO", {
      style: "currency",
      currency: "COP",
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(value)
  }

  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-64">
        <Header
          title="Predicciones"
          description="Realiza predicciones con los modelos entrenados"
        />
        <div className="p-6 space-y-6">
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Input Form */}
            <div className="lg:col-span-2">
              <Card className="border-border bg-card">
                <CardHeader>
                  <CardTitle className="text-foreground">Datos de Entrada</CardTitle>
                  <CardDescription>
                    Ingresa los indicadores financieros para predecir el estado
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid gap-6 md:grid-cols-2">
                    {/* Ingresos Totales */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Ingresos Totales
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.ingresos_totales]}
                          onValueChange={([v]) =>
                            setValues({ ...values, ingresos_totales: v })
                          }
                          min={10000000}
                          max={500000000}
                          step={1000000}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {formatCurrency(values.ingresos_totales)}
                        </span>
                      </div>
                    </div>

                    {/* Gastos Personal */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Gastos Personal
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.gastos_personal]}
                          onValueChange={([v]) =>
                            setValues({ ...values, gastos_personal: v })
                          }
                          min={5000000}
                          max={400000000}
                          step={1000000}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {formatCurrency(values.gastos_personal)}
                        </span>
                      </div>
                    </div>

                    {/* Liquidez */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Ratio de Liquidez
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.liquidez]}
                          onValueChange={([v]) =>
                            setValues({ ...values, liquidez: v })
                          }
                          min={0.1}
                          max={3}
                          step={0.01}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {values.liquidez.toFixed(2)}
                        </span>
                      </div>
                    </div>

                    {/* Dias Efectivo */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Dias de Efectivo
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.dias_efectivo]}
                          onValueChange={([v]) =>
                            setValues({ ...values, dias_efectivo: v })
                          }
                          min={0}
                          max={120}
                          step={1}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {values.dias_efectivo} dias
                        </span>
                      </div>
                    </div>

                    {/* CFO */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Cash Flow Operativo
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.cfo]}
                          onValueChange={([v]) =>
                            setValues({ ...values, cfo: v })
                          }
                          min={-50000000}
                          max={100000000}
                          step={1000000}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {formatCurrency(values.cfo)}
                        </span>
                      </div>
                    </div>

                    {/* Endeudamiento */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Endeudamiento
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.endeudamiento]}
                          onValueChange={([v]) =>
                            setValues({ ...values, endeudamiento: v })
                          }
                          min={0}
                          max={1}
                          step={0.01}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {(values.endeudamiento * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    {/* GP Ratio */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Gastos/Ingresos Ratio
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.gp_ratio]}
                          onValueChange={([v]) =>
                            setValues({ ...values, gp_ratio: v })
                          }
                          min={0.3}
                          max={1.2}
                          step={0.01}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {(values.gp_ratio * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>

                    {/* HHI Fuentes */}
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">
                        Indice HHI
                      </label>
                      <div className="flex items-center gap-2">
                        <Slider
                          value={[values.hhi_fuentes]}
                          onValueChange={([v]) =>
                            setValues({ ...values, hhi_fuentes: v })
                          }
                          min={0}
                          max={1}
                          step={0.01}
                          className="flex-1"
                        />
                        <span className="w-24 text-right text-sm text-muted-foreground">
                          {values.hhi_fuentes.toFixed(2)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <Button
                      onClick={handlePredict}
                      disabled={isLoading}
                      className="flex-1"
                    >
                      {isLoading ? (
                        <>
                          <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          Procesando...
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Predecir
                        </>
                      )}
                    </Button>
                    <Button variant="outline" onClick={handleReset}>
                      <RotateCcw className="mr-2 h-4 w-4" />
                      Reset
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Results */}
            <div className="space-y-6">
              {result ? (
                <>
                  {/* Random Forest Result */}
                  <Card className="border-chart-1/30 bg-chart-1/5">
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <TreePine className="h-5 w-5 text-chart-1" />
                        <CardTitle className="text-foreground">Random Forest</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between">
                        <Badge
                          className={
                            result.randomForest.prediction === 0
                              ? "bg-accent/20 text-accent"
                              : "bg-destructive/20 text-destructive"
                          }
                        >
                          {result.randomForest.prediction === 0 ? (
                            <>
                              <CheckCircle2 className="mr-1 h-3 w-3" />
                              Estable
                            </>
                          ) : (
                            <>
                              <AlertTriangle className="mr-1 h-3 w-3" />
                              Critico
                            </>
                          )}
                        </Badge>
                        <span className="text-2xl font-bold text-foreground">
                          {(result.randomForest.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-3 h-2 overflow-hidden rounded-full bg-secondary">
                        <div
                          className={`h-full transition-all ${
                            result.randomForest.prediction === 0
                              ? "bg-accent"
                              : "bg-destructive"
                          }`}
                          style={{
                            width: `${result.randomForest.probability * 100}%`,
                          }}
                        />
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">
                        Probabilidad de situacion{" "}
                        {result.randomForest.prediction === 0 ? "estable" : "critica"}
                      </p>
                    </CardContent>
                  </Card>

                  {/* Gradient Boosting Result */}
                  <Card className="border-chart-2/30 bg-chart-2/5">
                    <CardHeader className="pb-3">
                      <div className="flex items-center gap-2">
                        <Zap className="h-5 w-5 text-chart-2" />
                        <CardTitle className="text-foreground">Gradient Boosting</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between">
                        <Badge
                          className={
                            result.gradientBoosting.prediction === 0
                              ? "bg-accent/20 text-accent"
                              : "bg-destructive/20 text-destructive"
                          }
                        >
                          {result.gradientBoosting.prediction === 0 ? (
                            <>
                              <CheckCircle2 className="mr-1 h-3 w-3" />
                              Estable
                            </>
                          ) : (
                            <>
                              <AlertTriangle className="mr-1 h-3 w-3" />
                              Critico
                            </>
                          )}
                        </Badge>
                        <span className="text-2xl font-bold text-foreground">
                          {(result.gradientBoosting.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-3 h-2 overflow-hidden rounded-full bg-secondary">
                        <div
                          className={`h-full transition-all ${
                            result.gradientBoosting.prediction === 0
                              ? "bg-accent"
                              : "bg-destructive"
                          }`}
                          style={{
                            width: `${result.gradientBoosting.probability * 100}%`,
                          }}
                        />
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">
                        Probabilidad de situacion{" "}
                        {result.gradientBoosting.prediction === 0 ? "estable" : "critica"}
                      </p>
                    </CardContent>
                  </Card>

                  {/* Consensus */}
                  <Card className="border-border bg-card">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-foreground">Consenso</CardTitle>
                    </CardHeader>
                    <CardContent>
                      {result.randomForest.prediction ===
                      result.gradientBoosting.prediction ? (
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-5 w-5 text-accent" />
                          <span className="text-sm text-foreground">
                            Ambos modelos coinciden en la prediccion
                          </span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="h-5 w-5 text-chart-3" />
                          <span className="text-sm text-foreground">
                            Los modelos tienen predicciones diferentes
                          </span>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Card className="border-border bg-card">
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <div className="mb-4 rounded-full bg-secondary p-4">
                      <Play className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <p className="text-sm text-muted-foreground text-center">
                      Ajusta los parametros y presiona "Predecir" para obtener
                      los resultados de los modelos
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
