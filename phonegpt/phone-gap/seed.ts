// supabase 去做向量化的知识库数据
import { createOpenAI } from "@ai-sdk/openai";
// langchain  loader 是 RAG的基础功能 txt,pdf,excel....
// 加载网页内容
import {
    PuppeteerWebBaseLoader
} from '@langchain/community/document_loaders/web/puppeteer'
import {
    RecursiveCharacterTextSplitter
} from 'langchain/text_splitter'
import {
    embed // 向量嵌入
} from 'ai'
import "dotenv/config"
import { createClient } from "@supabase/supabase-js"


const supabase = createClient(
    process.env.SUPABASE_URL ?? "",
    process.env.SUPABASE_KEY ?? ""
)

const openai = createOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_API_BASE_URL
})


console.log('开始向量化蝴蝶知识库数据');
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512, // 切割的长度 包含一个比较独立的语义
    chunkOverlap: 100, // 切割的重叠长度 为了避免切割的句子中间被截断 100字符
})
const scrapePage = async (url: string): Promise<string> => {
    const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: {
            executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
            headless: true,

        },
        gotoOptions: {
            waitUntil: 'domcontentloaded',
        },
        evaluate: async (page, browser) => {
            const result = await page.evaluate(() => document.body.innerHTML);
            await browser.close();
            return result;
        }
    });
    // gm 正则修饰符
    // ^在[^]表示不是>的字符
    return (await loader.scrape()).replace(/<[^>]*>?/gm, "");
}
const loadData = async (webpages: string[]) => {
    let totalChunks = 0;
    const maxChunks = 150; // 限制最多处理150条信息

    for (const url of webpages) {
        if (totalChunks >= maxChunks) {
            console.log(`已达到最大处理数量 ${maxChunks} 条，停止处理`);
            break;
        }

        console.log(`正在处理网页: ${url}`);
        const content = await scrapePage(url);
        const chunks = await splitter.splitText(content);
        console.log(`从 ${url} 获取到 ${chunks.length} 个文本块`);

        for (let chunk of chunks) {
            if (totalChunks >= maxChunks) {
                console.log(`已达到最大处理数量 ${maxChunks} 条，停止处理当前页面`);
                break;
            }

            const { embedding } = await embed({
                model: openai.embedding('text-embedding-3-small'),
                value: chunk,
            })

            const { error } = await supabase.from("chunks").insert({
                content: chunk,
                vector: embedding,
                url,
            })

            if (error) {
                console.error('插入数据时出错:', error);
            } else {
                totalChunks++;
                console.log(`已处理 ${totalChunks}/${maxChunks} 条蝴蝶信息`);
            }
        }
    }

    console.log(`蝴蝶知识库构建完成，总共处理了 ${totalChunks} 条信息`);
}

// 维护一个蝴蝶知识库，知识库的来源可配置
loadData([
    "https://en.wikipedia.org/wiki/Butterfly",
    "https://en.wikipedia.org/wiki/Lepidoptera",
    "https://en.wikipedia.org/wiki/Monarch_butterfly",
    "https://en.wikipedia.org/wiki/Swallowtail_butterfly",
    "https://en.wikipedia.org/wiki/Butterfly_life_cycle",
    "https://en.wikipedia.org/wiki/Butterfly_migration",
    "https://en.wikipedia.org/wiki/List_of_butterflies_of_North_America",
    "https://en.wikipedia.org/wiki/Butterfly_conservation",
]);