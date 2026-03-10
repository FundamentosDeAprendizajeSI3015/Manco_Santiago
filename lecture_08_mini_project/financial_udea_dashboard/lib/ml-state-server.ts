import type { MLState } from "./ml-store"

// In-memory state storage (persists across requests in the same server instance)
let mlState: MLState = {
  datasetUploaded: false,
  datasetName: null,
  isTraining: false,
  trainingComplete: false,
  eda: null,
  training: null,
  error: null,
  csvPreview: null,
}

export function getMLState(): MLState {
  return mlState
}

export function updateMLState(newState: Partial<MLState>): void {
  mlState = { ...mlState, ...newState }
}

export function resetMLState(): void {
  mlState = {
    datasetUploaded: false,
    datasetName: null,
    isTraining: false,
    trainingComplete: false,
    eda: null,
    training: null,
    error: null,
    csvPreview: null,
  }
}
