import { NextRequest, NextResponse } from "next/server"
import { writeFile, mkdir } from "fs/promises"
import { join } from "path"
import { updateMLState, getMLState } from "@/lib/ml-state-server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File | null

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    if (!file.name.endsWith(".csv")) {
      return NextResponse.json(
        { error: "Only CSV files are supported" },
        { status: 400 }
      )
    }

    // Read file content
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const csvContent = buffer.toString("utf-8")

    // Parse CSV preview (first 10 rows)
    const lines = csvContent.split("\n").filter((line) => line.trim())
    const csvPreview = lines.slice(0, 11).map((line) => {
      // Handle quoted fields
      const fields: string[] = []
      let current = ""
      let inQuotes = false

      for (const char of line) {
        if (char === '"') {
          inQuotes = !inQuotes
        } else if (char === "," && !inQuotes) {
          fields.push(current.trim())
          current = ""
        } else {
          current += char
        }
      }
      fields.push(current.trim())
      return fields
    })

    // Save file to temp directory
    const uploadsDir = join(process.cwd(), "tmp", "uploads")
    await mkdir(uploadsDir, { recursive: true })

    const filePath = join(uploadsDir, "dataset.csv")
    await writeFile(filePath, buffer)

    // Update state
    updateMLState({
      datasetUploaded: true,
      datasetName: file.name,
      csvPreview,
      isTraining: false,
      trainingComplete: false,
      eda: null,
      training: null,
      error: null,
    })

    return NextResponse.json({
      success: true,
      fileName: file.name,
      rows: lines.length - 1,
      columns: csvPreview[0]?.length || 0,
    })
  } catch (error) {
    console.error("Upload error:", error)
    return NextResponse.json(
      { error: "Failed to upload file" },
      { status: 500 }
    )
  }
}
