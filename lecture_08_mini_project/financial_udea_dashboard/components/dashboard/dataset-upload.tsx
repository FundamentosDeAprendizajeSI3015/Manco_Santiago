"use client"

import { useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FileSpreadsheet, CheckCircle2, AlertCircle, Loader2, Play, RotateCcw } from "lucide-react"
import { cn } from "@/lib/utils"
import { useMLStore } from "@/lib/ml-store"

export function DatasetUpload() {
  const {
    datasetUploaded,
    datasetName,
    isTraining,
    trainingComplete,
    csvPreview,
    error,
    uploadDataset,
    startTraining,
    resetState,
  } = useMLStore()

  const [isDragOver, setIsDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [useGridSearch, setUseGridSearch] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const processFile = async (file: File) => {
    if (!file.name.endsWith(".csv")) {
      return
    }

    setUploading(true)
    setUploadProgress(0)

    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => Math.min(prev + 15, 90))
    }, 100)

    await uploadDataset(file)

    clearInterval(progressInterval)
    setUploadProgress(100)

    setTimeout(() => {
      setUploading(false)
      setUploadProgress(0)
    }, 300)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      processFile(file)
    }
  }, [])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      processFile(file)
    }
  }

  const handleTrain = async () => {
    await startTraining(useGridSearch)
  }

  const handleReset = async () => {
    await resetState()
  }

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle className="text-foreground">Subir Dataset</CardTitle>
        <CardDescription>
          Sube tu archivo CSV con datos financieros para entrenar los modelos
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={cn(
            "relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
            isDragOver
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-secondary/50",
            datasetUploaded && "border-accent bg-accent/5",
            error && "border-destructive bg-destructive/5",
            (uploading || isTraining) && "pointer-events-none opacity-60"
          )}
        >
          {!datasetUploaded && !uploading && (
            <>
              <div className="mb-4 rounded-full bg-secondary p-4">
                <Upload className="h-8 w-8 text-muted-foreground" />
              </div>
              <p className="mb-2 text-sm font-medium text-foreground">
                Arrastra tu archivo CSV aqui
              </p>
              <p className="mb-4 text-xs text-muted-foreground">
                o haz click para seleccionar
              </p>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileInput}
                className="absolute inset-0 cursor-pointer opacity-0"
              />
              <Button variant="outline" size="sm">
                Seleccionar Archivo
              </Button>
            </>
          )}

          {uploading && (
            <>
              <Loader2 className="mb-4 h-8 w-8 animate-spin text-primary" />
              <p className="mb-2 text-sm font-medium text-foreground">
                Subiendo dataset...
              </p>
              <div className="w-full max-w-xs">
                <div className="mb-2 flex justify-between text-xs text-muted-foreground">
                  <span>Procesando...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            </>
          )}

          {datasetUploaded && !uploading && (
            <>
              <div className="mb-4 rounded-full bg-accent/20 p-4">
                <CheckCircle2 className="h-8 w-8 text-accent" />
              </div>
              <p className="mb-2 text-sm font-medium text-foreground">
                Dataset cargado exitosamente
              </p>
              <div className="mb-4 flex items-center gap-2 rounded-lg bg-secondary px-4 py-2">
                <FileSpreadsheet className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-foreground">{datasetName}</span>
              </div>
            </>
          )}
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-lg bg-destructive/10 p-3 text-destructive">
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {datasetUploaded && (
          <div className="mt-6 space-y-4">
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  checked={useGridSearch}
                  onChange={(e) => setUseGridSearch(e.target.checked)}
                  className="rounded border-border"
                  disabled={isTraining || trainingComplete}
                />
                Usar GridSearchCV (mas lento pero mejor optimizacion)
              </label>
            </div>

            <div className="flex gap-3">
              <Button
                onClick={handleTrain}
                disabled={isTraining || trainingComplete}
                className="flex-1"
              >
                {isTraining ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Entrenando modelos...
                  </>
                ) : trainingComplete ? (
                  <>
                    <CheckCircle2 className="mr-2 h-4 w-4" />
                    Entrenamiento completado
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Iniciar Entrenamiento
                  </>
                )}
              </Button>

              <Button variant="outline" onClick={handleReset} disabled={isTraining}>
                <RotateCcw className="mr-2 h-4 w-4" />
                Reiniciar
              </Button>
            </div>
          </div>
        )}

        {csvPreview && csvPreview.length > 0 && (
          <div className="mt-6">
            <h4 className="mb-3 text-sm font-medium text-foreground">
              Vista previa del dataset
            </h4>
            <div className="overflow-x-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-secondary/50">
                    {csvPreview[0]?.map((header, i) => (
                      <th
                        key={i}
                        className="px-3 py-2 text-left text-xs font-medium text-muted-foreground"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csvPreview.slice(1, 6).map((row, i) => (
                    <tr key={i} className="border-b border-border/50 last:border-0">
                      {row.map((cell, j) => (
                        <td key={j} className="px-3 py-2 text-xs text-foreground">
                          {cell.length > 15 ? cell.slice(0, 15) + "..." : cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {csvPreview.length > 6 && (
              <p className="mt-2 text-center text-xs text-muted-foreground">
                Mostrando 5 de {csvPreview.length - 1} filas
              </p>
            )}
          </div>
        )}

        {!csvPreview && (
          <div className="mt-6">
            <h4 className="mb-3 text-sm font-medium text-foreground">
              Columnas Requeridas
            </h4>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
              {[
                "ingresos_totales",
                "gastos_personal",
                "liquidez",
                "dias_efectivo",
                "cfo",
                "endeudamiento",
                "gp_ratio",
                "label",
              ].map((col) => (
                <div
                  key={col}
                  className="rounded-md bg-secondary px-3 py-1.5 text-xs font-mono text-muted-foreground"
                >
                  {col}
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
