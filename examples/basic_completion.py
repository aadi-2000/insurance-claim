"""
Example: Basic LLM Completion
Demonstrates how to use the GPT client for simple text completion.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.gpt_client import GPTClient


async def main():
    client = GPTClient()

    # Simple completion
    messages = [{"role": "user", "content": "Explain insurance claim pre-screening in 2 sentences."}]
    result = await client.complete(messages)
    print("Response:", result["content"])
    print("Tokens used:", result["usage"]["total_tokens"])

    # JSON completion
    messages = [{"role": "user", "content": "List 3 common insurance fraud types with risk levels."}]
    result = await client.complete_json(messages)
    print("\nJSON Response:", result.get("parsed"))


if __name__ == "__main__":
    asyncio.run(main())
