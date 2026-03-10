import { NextResponse } from "next/server"
import { resetMLState } from "@/lib/ml-state-server"

export async function POST() {
  resetMLState()
  return NextResponse.json({ success: true })
}
