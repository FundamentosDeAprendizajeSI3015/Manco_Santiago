"use client"

import { Sidebar } from "@/components/dashboard/sidebar"
import { Header } from "@/components/dashboard/header"
import { DatasetUpload } from "@/components/dashboard/dataset-upload"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { FileSpreadsheet, AlertCircle, CheckCircle2 } from "lucide-react"
import { DataTable } from "@/components/dashboard/data-table"

const expectedColumns = [
  { name: "unidad", type: "string", description: "Nombre de la unidad academica" },
  { name: "anio", type: "int", description: "Anio del registro" },
  { name: "ingresos_totales", type: "float", description: "Total de ingresos en COP" },
  { name: "gastos_personal", type: "float", description: "Gastos en personal" },
  { name: "liquidez", type: "float", description: "Ratio de liquidez corriente" },
  { name: "dias_efectivo", type: "int", description: "Dias de efectivo disponible" },
  { name: "cfo", type: "float", description: "Cash Flow Operativo" },
  { name: "participacion_ley30", type: "float", description: "% recursos Ley 30" },
  { name: "participacion_regalias", type: "float", description: "% recursos regalias" },
  { name: "participacion_servicios", type: "float", description: "% ingresos por servicios" },
  { name: "participacion_matriculas", type: "float", description: "% ingresos matriculas" },
  { name: "hhi_fuentes", type: "float", description: "Indice HHI de diversificacion" },
  { name: "endeudamiento", type: "float", description: "Ratio de endeudamiento" },
  { name: "tendencia_ingresos", type: "float", description: "Tendencia 3 anios" },
  { name: "gp_ratio", type: "float", description: "Gastos/Ingresos ratio" },
  { name: "label", type: "int", description: "0=Estable, 1=Critico" },
]

export default function UploadPage() {
  return (
    <div className="min-h-screen bg-background">
      <Sidebar />
      <main className="pl-64">
        <Header
          title="Subir Dataset"
          description="Carga tus datos financieros para entrenar los modelos"
        />
        <div className="p-6 space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Upload Component */}
            <DatasetUpload />

            {/* Instructions */}
            <Card className="border-border bg-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-foreground">
                  <FileSpreadsheet className="h-5 w-5" />
                  Formato Esperado
                </CardTitle>
                <CardDescription>
                  Tu archivo CSV debe seguir esta estructura
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="rounded-lg bg-secondary/50 p-4">
                  <h4 className="mb-2 text-sm font-medium text-foreground">
                    Requisitos del archivo:
                  </h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 text-accent shrink-0" />
                      Formato CSV con separador de comas
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 text-accent shrink-0" />
                      Primera fila debe contener los nombres de las columnas
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 text-accent shrink-0" />
                      Valores numericos sin formato de moneda
                    </li>
                    <li className="flex items-start gap-2">
                      <CheckCircle2 className="mt-0.5 h-4 w-4 text-accent shrink-0" />
                      Columna "label" con valores 0 o 1
                    </li>
                  </ul>
                </div>

                <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
                  <h4 className="mb-2 flex items-center gap-2 text-sm font-medium text-destructive">
                    <AlertCircle className="h-4 w-4" />
                    Importante
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Los valores faltantes seran imputados con la mediana para
                    variables numericas y "Unknown" para categoricas.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Column Reference */}
          <Card className="border-border bg-card">
            <CardHeader>
              <CardTitle className="text-foreground">Referencia de Columnas</CardTitle>
              <CardDescription>
                Descripcion detallada de cada variable esperada
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="pb-3 text-left font-medium text-muted-foreground">
                        Columna
                      </th>
                      <th className="pb-3 text-left font-medium text-muted-foreground">
                        Tipo
                      </th>
                      <th className="pb-3 text-left font-medium text-muted-foreground">
                        Descripcion
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {expectedColumns.map((col) => (
                      <tr
                        key={col.name}
                        className="border-b border-border/50 hover:bg-secondary/50"
                      >
                        <td className="py-3">
                          <code className="rounded bg-secondary px-2 py-0.5 text-xs font-mono text-primary">
                            {col.name}
                          </code>
                        </td>
                        <td className="py-3 text-muted-foreground">{col.type}</td>
                        <td className="py-3 text-foreground">{col.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Data Preview after training */}
          <DataTable />
        </div>
      </main>
    </div>
  )
}
