#!/usr/bin/env python3
"""
Debug script to test OCR and LLM extraction on the discharge summary
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.image_agent import ImageProcessingAgent
from agents.requirements_agent import RequirementsAgent
from llm.gpt_client import GPTClient
import os
from dotenv import load_dotenv

load_dotenv()

async def test_extraction():
    # Initialize LLM client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No OpenAI API key found in .env")
        return
    
    llm_client = GPTClient(api_key=api_key)
    print("✅ LLM client initialized")
    
    # Initialize agents
    image_agent = ImageProcessingAgent(llm_client=llm_client)
    requirements_agent = RequirementsAgent(llm_client=llm_client)
    
    # Load test image (you'll need to provide the path)
    image_path = "test_discharge_summary.png"  # Update this path
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("Please save the discharge summary image as 'test_discharge_summary.png'")
        return
    
    with open(image_path, 'rb') as f:
        file_bytes = f.read()
    
    print(f"\n{'='*80}")
    print("TESTING IMAGE EXTRACTION")
    print('='*80)
    
    # Test image agent
    result = await image_agent.process(file_bytes, image_path, "image/png")
    
    extracted_text = result['output']['extracted_text']
    print(f"\n📄 Extracted Text ({len(extracted_text)} characters):")
    print("-" * 80)
    print(extracted_text[:500])  # First 500 chars
    print("-" * 80)
    
    print(f"\n{'='*80}")
    print("TESTING REQUIREMENTS EXTRACTION")
    print('='*80)
    
    # Test requirements agent
    req_result = await requirements_agent.process(
        image_data=result,
        pdf_data={}
    )
    
    print(f"\n📋 Extracted Fields:")
    for field, value in req_result['output']['extracted_requirements'].items():
        status = "✅" if value else "❌"
        print(f"  {status} {field}: {value}")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"Requirements Met: {req_result['output']['requirements_met']}")
    print(f"Missing Fields: {req_result['output']['missing_fields']}")

if __name__ == "__main__":
    asyncio.run(test_extraction())
