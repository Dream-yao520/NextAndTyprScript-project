import {
    embed,
    streamText
} from 'ai';
import {
    createOpenAI
} from '@ai-sdk/openai';
import {
    createClient
} from '@supabase/supabase-js';

interface ChatChunk {
    url: string;
    date_updated: string;
    content: string;
}

const supabase = createClient(
    process.env.SUPABASE_URL ?? "",
    process.env.SUPABASE_KEY ?? ""
);

const openai = createOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_API_BASE_URL,
})

async function generateEmbedding(message: string) {
    return embed({
        model: openai.embedding('text-embedding-3-small'),
        value: message
    })
}

async function fetchRelevantContext(embedding: number[]) {
    const {
        data,
        error
    } = await supabase.rpc("get_relevant_chunks", {
        query_vector: embedding,
        match_threshold: 0.7,
        match_count: 3
    })

    if (error) throw error;
    console.log(data, '////////////////')
    return JSON.stringify(
        data.map((item: ChatChunk) => `
        Source: ${item.url},
        Date Updated: ${item.date_updated}
        Content: ${item.content}  
      `)
    )
}


const createPrompt = (context: string, userQuestion: string) => {
    return {
        role: "system",
        content: `
            你是一个专门提供蝴蝶相关信息的智能助手。
      请使用以下上下文信息来回答问题：
      ----------------
      开始上下文
      ${context}
      结束上下文
      ----------------
      
      请用markdown格式返回答案，包含相关链接和信息最后更新的日期。
      如果上述上下文信息不足以回答问题，请基于你的知识提供答案，但要提醒用户这些信息可能不是最新的。
      如果用户问的问题与蝴蝶无关，请礼貌地告知你只能回答蝴蝶相关的问题。
      
      ----------------
      问题: ${userQuestion}
      ----------------
        `
    }
}

export async function POST(req: Request) {
    try {
        const { messages } = await req.json();
        const latestMessage = messages.at(-1).content;
        // embedding
        const { embedding } = await generateEmbedding(latestMessage);
        // console.log(embedding);
        // 相似度计算
        const context = await fetchRelevantContext(embedding);
        const prompt = createPrompt(context, latestMessage)
        console.log(prompt)
        const result = streamText({
            model: openai("gpt-4o-mini"),
            messages: [prompt, ...messages]
        })
        return result.toDataStreamResponse()
    } catch (err) {
        throw err
    }
}