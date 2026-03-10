import { NextResponse } from "next/server"
import { getMLState } from "@/lib/ml-state-server"

export async function GET() {
  const state = getMLState()
  return NextResponse.json(state)
}
