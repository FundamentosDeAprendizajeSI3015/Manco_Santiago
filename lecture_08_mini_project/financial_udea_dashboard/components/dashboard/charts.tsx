"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { useMLStore, type ModelResult, type EDAStats } from "@/lib/ml-store"

interface ChartProps {
  modelType?: "random_forest" | "gradient_boosting"
  dataSet?: "train" | "validation" | "test"
}

export function ROCCurveChart({ dataSet = "test" }: ChartProps) {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Curva ROC</CardTitle>
          <CardDescription>Comparacion Random Forest vs Gradient Boosting</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver la curva ROC
          </div>
        </CardContent>
      </Card>
    )
  }

  const rfData = training.random_forest[dataSet].roc_curve
  const gbData = training.gradient_boosting[dataSet].roc_curve

  // Combine ROC curve data
  const rocCurveData = rfData.fpr.map((fpr, i) => ({
    fpr,
    rf: rfData.tpr[i] || 0,
    gb: gbData.tpr[i] || 0,
  }))

  const rfAuc = training.random_forest[dataSet].auc.toFixed(2)
  const gbAuc = training.gradient_boosting[dataSet].auc.toFixed(2)

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Curva ROC</CardTitle>
        <CardDescription>Comparacion Random Forest vs Gradient Boosting ({dataSet})</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={rocCurveData}>
            <defs>
              <linearGradient id="rfGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--chart-1)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="var(--chart-1)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gbGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--chart-2)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="var(--chart-2)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis 
              dataKey="fpr" 
              stroke="var(--muted-foreground)" 
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <YAxis 
              stroke="var(--muted-foreground)" 
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="fpr" 
              stroke="var(--muted-foreground)" 
              strokeDasharray="5 5"
              name="Random (baseline)"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="rf"
              stroke="var(--chart-1)"
              fill="url(#rfGradient)"
              name={`Random Forest (AUC: ${rfAuc})`}
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="gb"
              stroke="var(--chart-2)"
              fill="url(#gbGradient)"
              name={`Gradient Boosting (AUC: ${gbAuc})`}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export function FeatureImportanceChart({ modelType }: ChartProps) {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Importancia de Variables</CardTitle>
          <CardDescription>Top features para clasificacion financiera</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver la importancia de variables
          </div>
        </CardContent>
      </Card>
    )
  }

  // Combine feature importance from both models
  const rfFeatures = training.random_forest.feature_importance
  const gbFeatures = training.gradient_boosting.feature_importance

  const featureMap = new Map<string, { rf: number; gb: number }>()
  
  rfFeatures.forEach(({ feature, importance }) => {
    featureMap.set(feature, { rf: importance, gb: 0 })
  })
  
  gbFeatures.forEach(({ feature, importance }) => {
    const existing = featureMap.get(feature)
    if (existing) {
      existing.gb = importance
    } else {
      featureMap.set(feature, { rf: 0, gb: importance })
    }
  })

  const featureImportanceData = Array.from(featureMap.entries())
    .map(([feature, values]) => ({ feature, ...values }))
    .sort((a, b) => (b.rf + b.gb) - (a.rf + a.gb))
    .slice(0, 10)

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Importancia de Variables</CardTitle>
        <CardDescription>Top features para clasificacion financiera</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={featureImportanceData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
            <XAxis 
              type="number" 
              stroke="var(--muted-foreground)"
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <YAxis 
              dataKey="feature" 
              type="category" 
              stroke="var(--muted-foreground)"
              tickLine={false}
              axisLine={false}
              width={130}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Legend />
            <Bar dataKey="rf" fill="var(--chart-1)" name="Random Forest" radius={[0, 4, 4, 0]} />
            <Bar dataKey="gb" fill="var(--chart-2)" name="Gradient Boosting" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export function MetricsComparisonChart({ dataSet = "test" }: ChartProps) {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Comparacion de Metricas</CardTitle>
          <CardDescription>Random Forest vs Gradient Boosting</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver las metricas
          </div>
        </CardContent>
      </Card>
    )
  }

  const rfMetrics = training.random_forest[dataSet]
  const gbMetrics = training.gradient_boosting[dataSet]

  const metricsComparisonData = [
    { metric: "Accuracy", rf: rfMetrics.accuracy, gb: gbMetrics.accuracy },
    { metric: "Precision", rf: rfMetrics.precision, gb: gbMetrics.precision },
    { metric: "Recall", rf: rfMetrics.recall, gb: gbMetrics.recall },
    { metric: "F1 Score", rf: rfMetrics.f1, gb: gbMetrics.f1 },
    { metric: "AUC", rf: rfMetrics.auc, gb: gbMetrics.auc },
  ]

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Comparacion de Metricas</CardTitle>
        <CardDescription>Random Forest vs Gradient Boosting en {dataSet}</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={metricsComparisonData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis 
              dataKey="metric" 
              stroke="var(--muted-foreground)"
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="var(--muted-foreground)"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Legend />
            <Bar dataKey="rf" fill="var(--chart-1)" name="Random Forest" radius={[4, 4, 0, 0]} />
            <Bar dataKey="gb" fill="var(--chart-2)" name="Gradient Boosting" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export function ConfusionMatrixChart({ modelType = "random_forest", dataSet = "test" }: ChartProps) {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Matriz de Confusion</CardTitle>
          <CardDescription>Predicciones vs Valores Reales</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[200px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver la matriz de confusion
          </div>
        </CardContent>
      </Card>
    )
  }

  const model = training[modelType]
  const cm = model[dataSet].confusion_matrix
  const modelName = modelType === "random_forest" ? "Random Forest" : "Gradient Boosting"

  const confusionData = [
    { name: "True Neg", value: cm[0]?.[0] || 0 },
    { name: "False Pos", value: cm[0]?.[1] || 0 },
    { name: "False Neg", value: cm[1]?.[0] || 0 },
    { name: "True Pos", value: cm[1]?.[1] || 0 },
  ]

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Matriz de Confusion</CardTitle>
        <CardDescription>{modelName} - {dataSet}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-2">
          {confusionData.map((cell, index) => (
            <div
              key={cell.name}
              className="flex flex-col items-center justify-center rounded-lg p-6"
              style={{ 
                backgroundColor: index === 0 || index === 3 ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)",
                border: `1px solid ${index === 0 || index === 3 ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)"}`
              }}
            >
              <span className="text-3xl font-bold text-foreground">{cell.value}</span>
              <span className="text-xs text-muted-foreground mt-1">{cell.name}</span>
            </div>
          ))}
        </div>
        <div className="mt-4 flex justify-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-accent" />
            <span className="text-muted-foreground">Prediccion Correcta</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-destructive" />
            <span className="text-muted-foreground">Prediccion Incorrecta</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function TargetDistributionChart() {
  const { eda, trainingComplete } = useMLStore()

  if (!trainingComplete || !eda) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Distribucion del Target</CardTitle>
          <CardDescription>Balance de clases en el dataset</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[200px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver la distribucion
          </div>
        </CardContent>
      </Card>
    )
  }

  const total = Object.values(eda.label_distribution).reduce((a, b) => a + b, 0)
  const targetDistribution = Object.entries(eda.label_distribution).map(([key, value]) => ({
    name: key === "0" ? "Estable" : "Critico",
    value: Math.round((value / total) * 100),
    fill: key === "0" ? "var(--chart-2)" : "var(--chart-4)",
  }))

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Distribucion del Target</CardTitle>
        <CardDescription>Balance de clases en el dataset</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={targetDistribution}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={80}
              paddingAngle={5}
              dataKey="value"
            >
              {targetDistribution.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${value}%`}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-2 flex justify-center gap-6 text-sm">
          {targetDistribution.map((item) => (
            <div key={item.name} className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
              <span className="text-muted-foreground">{item.name}: {item.value}%</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function TrainingHistoryChart() {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Metricas por Set</CardTitle>
          <CardDescription>Comparacion Train vs Validation vs Test</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver las metricas
          </div>
        </CardContent>
      </Card>
    )
  }

  const sets = ["train", "validation", "test"] as const
  const trainingHistory = sets.map((set) => ({
    set: set.charAt(0).toUpperCase() + set.slice(1),
    rf_acc: training.random_forest[set].accuracy,
    gb_acc: training.gradient_boosting[set].accuracy,
    rf_f1: training.random_forest[set].f1,
    gb_f1: training.gradient_boosting[set].f1,
  }))

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Metricas por Set</CardTitle>
        <CardDescription>Comparacion Train vs Validation vs Test</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trainingHistory}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis 
              dataKey="set" 
              stroke="var(--muted-foreground)"
              tickLine={false}
              axisLine={false}
            />
            <YAxis 
              stroke="var(--muted-foreground)"
              tickLine={false}
              axisLine={false}
              domain={[0, 1]}
              tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="rf_acc" 
              stroke="var(--chart-1)" 
              name="RF Accuracy"
              strokeWidth={2}
              dot={{ fill: "var(--chart-1)", strokeWidth: 0, r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="gb_acc" 
              stroke="var(--chart-2)" 
              name="GB Accuracy"
              strokeWidth={2}
              dot={{ fill: "var(--chart-2)", strokeWidth: 0, r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="rf_f1" 
              stroke="var(--chart-1)" 
              name="RF F1"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: "var(--chart-1)", strokeWidth: 0, r: 5 }}
            />
            <Line 
              type="monotone" 
              dataKey="gb_f1" 
              stroke="var(--chart-2)" 
              name="GB F1"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: "var(--chart-2)", strokeWidth: 0, r: 5 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export function ModelScoresRadial() {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Scores de Modelos</CardTitle>
          <CardDescription>Rendimiento general</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[200px] items-center justify-center text-muted-foreground">
            Entrena un modelo para ver los scores
          </div>
        </CardContent>
      </Card>
    )
  }

  const data = [
    { name: "GB Accuracy", value: Math.round(training.gradient_boosting.test.accuracy * 100), fill: "var(--chart-2)" },
    { name: "GB F1", value: Math.round(training.gradient_boosting.test.f1 * 100), fill: "var(--chart-2)" },
    { name: "RF Accuracy", value: Math.round(training.random_forest.test.accuracy * 100), fill: "var(--chart-1)" },
    { name: "RF F1", value: Math.round(training.random_forest.test.f1 * 100), fill: "var(--chart-1)" },
  ]

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Scores de Modelos</CardTitle>
        <CardDescription>Rendimiento general (Test Set)</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <RadialBarChart
            cx="50%"
            cy="50%"
            innerRadius="20%"
            outerRadius="90%"
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <RadialBar
              dataKey="value"
              cornerRadius={10}
              background={{ fill: "var(--secondary)" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${value}%`}
            />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="flex flex-wrap justify-center gap-4 text-xs">
          {data.map((item) => (
            <div key={item.name} className="flex items-center gap-1.5">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: item.fill }} />
              <span className="text-muted-foreground">{item.name}: {item.value}%</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function DataSplitChart() {
  const { training, trainingComplete } = useMLStore()

  if (!trainingComplete || !training) {
    return null
  }

  const splits = training.data_splits
  const total = splits.train + splits.validation + splits.test

  const data = [
    { name: "Train (60%)", value: splits.train, fill: "var(--chart-1)" },
    { name: "Validation (20%)", value: splits.validation, fill: "var(--chart-2)" },
    { name: "Test (20%)", value: splits.test, fill: "var(--chart-3)" },
  ]

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Division de Datos</CardTitle>
        <CardDescription>Train / Validation / Test Split</CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={50}
              outerRadius={70}
              paddingAngle={3}
              dataKey="value"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--popover)",
                border: "1px solid var(--border)",
                borderRadius: "8px",
                color: "var(--popover-foreground)",
              }}
              formatter={(value: number) => `${value} muestras`}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="mt-2 flex flex-wrap justify-center gap-4 text-xs">
          {data.map((item) => (
            <div key={item.name} className="flex items-center gap-2">
              <div className="h-3 w-3 rounded-full" style={{ backgroundColor: item.fill }} />
              <span className="text-muted-foreground">{item.name}: {item.value}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
