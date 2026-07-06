# Project Prompt History

This file is a Markdown version of the original prompt log from `prompts.txt`.

## Prompt 1

We are building: **Build a Memory First Web Agent in Python**.

Project description:

> Design and implement a GenAI agent that:
>
> 1. Answers user questions by checking a Redis memory first (vector search over embedded data).
> 2. Falls back to the web when memory misses, then ingests what it finds into Redis for future reuse.
> 3. Returns a grounded answer with clear metadata (URLs) to the sources used.
>
> You may implement using LangGraph, LangChain and Python.

### Memory-first routing

- Embed the user query, perform vector search in Redis.
- If top result similarity >= threshold (default `0.7`), answer from memory only (include stored metadata).
- Else search the web, fetch top pages, summarize, store chunks and metadata in Redis, then answer using retrieved context.

### Web search and fetch

- Use any web search APIs.
- Fetch web pages content after search.
- Convert webpages into markdown content.

### Log

- Log each turn: was it a memory hit or memory miss + web search.

### Analytics

- Provide analytics on what type of topics and questions users asked.

Additional instructions:

- The basic Python environment is already created.
- We are doing this in several steps.
- First, build a simple Python application that searches the web based on user query using LangGraph.
- Read LangGraph docs before implementation: https://docs.langchain.com/oss/python/langgraph/quickstart#use-the-graph-api
- For each web search, fetch top 3 pages, summarize them, then answer using the summary.
- Create the web search tool using Tavily API.
- Read Tavily docs: https://docs.tavily.com/sdk/python/quick-start
- We are using Azure OpenAI.

## Prompt 2

I am ready for step 2: memory-first routing with Redis vector search, threshold gate, ingestion on miss, plus hit/miss analytics dashboards.

## Prompt 3

I am plannig to push the solution to GitHub. Please, create `.gitignore` file.

## Prompt 4

Please, update `README.md` with instruction to run redis (docker). e.g.

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

## Prompt 5

The file `prompts.txt` contains all my prompts. Please, convert it to `PROMPTS.md` so it is nicer to read.

## Prompt 6

Modify README.md Memory-First Web Agent (Step 2) section to descibe all the functinalities (not only Step 2)

## Prompt 7

Please, add a short "Architrecture flow" diagram section (Mermaid)

## Prompt 8

Please, add a safeguard node on top of the graph. It should not allow basic prompt-injection. 

## Prompt 9

Please, update README.md. 
