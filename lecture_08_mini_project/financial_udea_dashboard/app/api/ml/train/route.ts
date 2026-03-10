import { NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import { join } from "path"
import { updateMLState, getMLState } from "@/lib/ml-state-server"

export async function POST(request: NextRequest) {
  try {
    const state = getMLState()

    if (!state.datasetUploaded) {
      return NextResponse.json(
        { error: "No dataset uploaded" },
        { status: 400 }
      )
    }

    const body = await request.json().catch(() => ({}))
    const useGridSearch = body.useGridSearch || false

    // Update state to training
    updateMLState({
      ...state,
      isTraining: true,
      trainingComplete: false,
      error: null,
    })

    const csvPath = join(process.cwd(), "tmp", "uploads", "dataset.csv")
    const scriptPath = join(process.cwd(), "scripts", "ml_training.py")

    // Run Python training script
    const args = [scriptPath, csvPath]
    if (useGridSearch) {
      args.push("--grid-search")
    }

    return new Promise<NextResponse>((resolve) => {
      const pythonProcess = spawn("py", args, {
        cwd: join(process.cwd(), "scripts"),
      })

      let stdout = ""
      let stderr = ""

      pythonProcess.stdout.on("data", (data) => {
        stdout += data.toString()
      })

      pythonProcess.stderr.on("data", (data) => {
        stderr += data.toString()
      })

      pythonProcess.on("close", (code) => {
        if (code !== 0) {
          console.error("Python error:", stderr)
          updateMLState({
            ...getMLState(),
            isTraining: false,
            trainingComplete: false,
            error: `Training failed: ${stderr || "Unknown error"}`,
          })
          resolve(
            NextResponse.json(
              { error: "Training failed", details: stderr },
              { status: 500 }
            )
          )
          return
        }

        try {
          const result = JSON.parse(stdout)

          if (!result.success) {
            updateMLState({
              ...getMLState(),
              isTraining: false,
              trainingComplete: false,
              error: result.error || "Training failed",
            })
            resolve(
              NextResponse.json({ error: result.error }, { status: 500 })
            )
            return
          }

          // Update state with results
          updateMLState({
            ...getMLState(),
            isTraining: false,
            trainingComplete: true,
            eda: result.eda,
            training: result.training,
            error: null,
          })

          resolve(NextResponse.json({ success: true, result }))
        } catch (parseError) {
          console.error("Parse error:", parseError, "stdout:", stdout)
          updateMLState({
            ...getMLState(),
            isTraining: false,
            trainingComplete: false,
            error: "Failed to parse training results",
          })
          resolve(
            NextResponse.json(
              { error: "Failed to parse results" },
              { status: 500 }
            )
          )
        }
      })
    })
  } catch (error) {
    console.error("Training error:", error)
    updateMLState({
      ...getMLState(),
      isTraining: false,
      error: "Training failed unexpectedly",
    })
    return NextResponse.json({ error: "Training failed" }, { status: 500 })
  }
}
