"""
Example: Chain-of-Thought Prompting for Claim Assessment
Demonstrates multi-step reasoning with the CoT builder.
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.gpt_client import GPTClient
from src.prompt_engineering.chain import ChainOfThoughtBuilder


async def main():
    client = GPTClient()
    cot = ChainOfThoughtBuilder()

    claim_data = {
        "patient": "Rajesh Kumar",
        "diagnosis": "STEMI",
        "procedure": "Coronary Angioplasty",
        "amount": 485000,
        "hospital": "Apollo Hospitals",
    }

    prompt = cot.build_claim_analysis_chain(claim_data)
    print("Chain-of-Thought Prompt:\n", prompt[:500], "...\n")

    messages = [{"role": "user", "content": f"{prompt}\n\nClaim data: {claim_data}"}]
    result = await client.complete(messages, max_tokens=1000)
    print("Analysis:\n", result["content"][:500])


if __name__ == "__main__":
    asyncio.run(main())
