import useSWR, { mutate } from "swr"

export interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1: number
  auc: number
  confusion_matrix: number[][]
  roc_curve: {
    fpr: number[]
    tpr: number[]
  }
}

export interface ModelResult {
  train: ModelMetrics
  validation: ModelMetrics
  test: ModelMetrics
  feature_importance: { feature: string; importance: number }[]
}

export interface TrainingResults {
  random_forest: ModelResult
  gradient_boosting: ModelResult
  best_params: {
    random_forest: Record<string, unknown>
    gradient_boosting: Record<string, unknown>
  }
  cv_scores: {
    random_forest: number
    gradient_boosting: number
  }
  data_splits: {
    train: number
    validation: number
    test: number
  }
}

export interface EDAStats {
  total_samples: number
  features_count: number
  label_distribution: Record<string, number>
  numeric_stats: Record<
    string,
    {
      mean: number
      median: number
      std: number
      min: number
      max: number
    }
  >
  correlation_matrix: Record<string, Record<string, number>>
}

export interface MLState {
  datasetUploaded: boolean
  datasetName: string | null
  isTraining: boolean
  trainingComplete: boolean
  eda: EDAStats | null
  training: TrainingResults | null
  error: string | null
  csvPreview: string[][] | null
}

const ML_STORE_KEY = "ml-training-state"

// Default empty state
const defaultState: MLState = {
  datasetUploaded: false,
  datasetName: null,
  isTraining: false,
  trainingComplete: false,
  eda: null,
  training: null,
  error: null,
  csvPreview: null,
}

// Fetcher that gets state from API
const fetcher = async (url: string): Promise<MLState> => {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error("Failed to fetch ML state")
  }
  return res.json()
}

// Custom hook for ML state
export function useMLStore() {
  const { data, error, isLoading } = useSWR<MLState>(
    "/api/ml/state",
    fetcher,
    {
      refreshInterval: 1000, // Poll every second during training
      revalidateOnFocus: true,
      fallbackData: defaultState,
    }
  )

  const state = data || defaultState

  const uploadDataset = async (file: File): Promise<boolean> => {
    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch("/api/ml/upload", {
        method: "POST",
        body: formData,
      })

      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.error || "Upload failed")
      }

      // Revalidate state
      await mutate("/api/ml/state")
      return true
    } catch (err) {
      console.error("Upload error:", err)
      return false
    }
  }

  const startTraining = async (useGridSearch = false): Promise<boolean> => {
    try {
      const res = await fetch("/api/ml/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ useGridSearch }),
      })

      if (!res.ok) {
        const errorData = await res.json()
        throw new Error(errorData.error || "Training failed")
      }

      // Revalidate state
      await mutate("/api/ml/state")
      return true
    } catch (err) {
      console.error("Training error:", err)
      return false
    }
  }

  const resetState = async (): Promise<void> => {
    await fetch("/api/ml/reset", { method: "POST" })
    await mutate("/api/ml/state")
  }

  return {
    ...state,
    isLoading,
    hasError: !!error,
    uploadDataset,
    startTraining,
    resetState,
    refresh: () => mutate("/api/ml/state"),
  }
}

// Export types for use in components
export type { MLState as MLStoreState }
