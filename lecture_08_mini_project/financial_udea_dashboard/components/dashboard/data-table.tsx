"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { useMLStore } from "@/lib/ml-store"

const formatCurrency = (value: number) => {
  return new Intl.NumberFormat("es-CO", {
    style: "currency",
    currency: "COP",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value)
}

export function DataTable() {
  const { eda, csvPreview, trainingComplete } = useMLStore()

  if (!trainingComplete || !eda || !csvPreview || csvPreview.length < 2) {
    return (
      <Card className="border-border bg-card">
        <CardHeader>
          <CardTitle className="text-foreground">Dataset FIRE_UdeA</CardTitle>
          <CardDescription>
            Vista previa de los datos financieros por unidad academica
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-[200px] items-center justify-center text-muted-foreground">
            Sube un dataset para ver los datos aqui
          </div>
        </CardContent>
      </Card>
    )
  }

  const headers = csvPreview[0]
  const rows = csvPreview.slice(1, 11) // First 10 rows

  // Find column indices
  const findColIndex = (name: string) => headers.findIndex(h => h.toLowerCase().includes(name.toLowerCase()))
  
  const unidadIdx = findColIndex("unidad")
  const anioIdx = findColIndex("anio")
  const ingresosIdx = findColIndex("ingresos")
  const gastosIdx = findColIndex("gastos")
  const liquidezIdx = findColIndex("liquidez")
  const cfoIdx = findColIndex("cfo")
  const endeudamientoIdx = findColIndex("endeudamiento")
  const labelIdx = findColIndex("label")

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Dataset FIRE_UdeA</CardTitle>
        <CardDescription>
          Vista previa de los datos financieros - {eda.total_samples} registros totales
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {unidadIdx >= 0 && <th className="pb-3 text-left font-medium text-muted-foreground">Unidad</th>}
                {anioIdx >= 0 && <th className="pb-3 text-left font-medium text-muted-foreground">Anio</th>}
                {ingresosIdx >= 0 && <th className="pb-3 text-right font-medium text-muted-foreground">Ingresos</th>}
                {gastosIdx >= 0 && <th className="pb-3 text-right font-medium text-muted-foreground">Gastos</th>}
                {liquidezIdx >= 0 && <th className="pb-3 text-right font-medium text-muted-foreground">Liquidez</th>}
                {cfoIdx >= 0 && <th className="pb-3 text-right font-medium text-muted-foreground">CFO</th>}
                {endeudamientoIdx >= 0 && <th className="pb-3 text-right font-medium text-muted-foreground">Endeud.</th>}
                {labelIdx >= 0 && <th className="pb-3 text-center font-medium text-muted-foreground">Estado</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, idx) => {
                const ingresos = ingresosIdx >= 0 ? parseFloat(row[ingresosIdx]) : 0
                const gastos = gastosIdx >= 0 ? parseFloat(row[gastosIdx]) : 0
                const liquidez = liquidezIdx >= 0 ? parseFloat(row[liquidezIdx]) : 0
                const cfo = cfoIdx >= 0 ? parseFloat(row[cfoIdx]) : 0
                const endeudamiento = endeudamientoIdx >= 0 ? parseFloat(row[endeudamientoIdx]) : 0
                const label = labelIdx >= 0 ? parseInt(row[labelIdx]) : 0

                return (
                  <tr key={idx} className="border-b border-border/50 hover:bg-secondary/50">
                    {unidadIdx >= 0 && (
                      <td className="py-3 text-foreground max-w-[150px] truncate">
                        {row[unidadIdx]}
                      </td>
                    )}
                    {anioIdx >= 0 && (
                      <td className="py-3 text-muted-foreground">{row[anioIdx]}</td>
                    )}
                    {ingresosIdx >= 0 && (
                      <td className="py-3 text-right text-foreground">
                        {formatCurrency(ingresos)}
                      </td>
                    )}
                    {gastosIdx >= 0 && (
                      <td className="py-3 text-right text-foreground">
                        {formatCurrency(gastos)}
                      </td>
                    )}
                    {liquidezIdx >= 0 && (
                      <td className={cn(
                        "py-3 text-right font-medium",
                        liquidez >= 1.2 ? "text-accent" : liquidez >= 1 ? "text-chart-3" : "text-destructive"
                      )}>
                        {liquidez.toFixed(2)}
                      </td>
                    )}
                    {cfoIdx >= 0 && (
                      <td className={cn(
                        "py-3 text-right font-medium",
                        cfo >= 0 ? "text-accent" : "text-destructive"
                      )}>
                        {formatCurrency(cfo)}
                      </td>
                    )}
                    {endeudamientoIdx >= 0 && (
                      <td className={cn(
                        "py-3 text-right font-medium",
                        endeudamiento <= 0.3 ? "text-accent" : endeudamiento <= 0.5 ? "text-chart-3" : "text-destructive"
                      )}>
                        {(endeudamiento * 100).toFixed(0)}%
                      </td>
                    )}
                    {labelIdx >= 0 && (
                      <td className="py-3 text-center">
                        <Badge
                          variant={label === 0 ? "default" : "destructive"}
                          className={cn(
                            label === 0
                              ? "bg-accent/20 text-accent hover:bg-accent/30"
                              : "bg-destructive/20 text-destructive hover:bg-destructive/30"
                          )}
                        >
                          {label === 0 ? "Estable" : "Critico"}
                        </Badge>
                      </td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        {rows.length < (csvPreview.length - 1) && (
          <p className="mt-4 text-center text-sm text-muted-foreground">
            Mostrando {rows.length} de {csvPreview.length - 1} filas
          </p>
        )}
      </CardContent>
    </Card>
  )
}
