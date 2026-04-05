"""
Example: Multi-turn Chat Session for Claim Analysis
Demonstrates conversational claim processing with GPT.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.gpt_client import GPTClient


async def main():
    client = GPTClient()
    history = []
    system = "You are an insurance claim analyst. Help the user understand their claim."

    queries = [
        "I had a cardiac procedure and need to file a claim. What documents do I need?",
        "My policy number is HLTH-2024-INS-77823. Can you check if cardiac procedures are covered?",
        "The total bill is ₹4,85,000. Does that seem reasonable for an angioplasty with stent?",
    ]

    for query in queries:
        history.append({"role": "user", "content": query})
        result = await client.complete(history, system_prompt=system)
        print(f"\nUser: {query}")
        print(f"Assistant: {result['content'][:200]}...")
        history.append({"role": "assistant", "content": result["content"]})


if __name__ == "__main__":
    asyncio.run(main())
