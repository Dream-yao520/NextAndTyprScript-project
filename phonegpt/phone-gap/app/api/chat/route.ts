import {
    embed,
    streamText
} from 'ai'
import {
    createOpenAI
} from '@ai-sdk/openai'
import { createClient } from "@supabase/supabase-js"

const supabase = createClient(
    process.env.SUPABASE_URL ?? "",
    process.env.SUPABASE_KEY ?? ""
)
const openai = createOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_API_BASE_URL
})
async function generateEmbedding(message: string) {
    return embed({
        model: openai.embedding('text-embedding-3-small'),
        value: message,
    })
}

export async function POST(req: Request) {
    try {
        const { messages } = await req.json()
        const lastestMessage = messages.at(-1).content
        // embedding
        const { embedding } = await generateEmbedding(lastestMessage)
        // console.log(embedding)
        // 相似度计算

    } catch (err) {

    }
}

async function fecthRelevantContext(embedding: number[]) {
    const {
        data,
        error
    } = await supabase.rpc("get_relevant_chunks", {

    })
}