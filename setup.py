from setuptools import setup, find_packages

setup(
    name="insurance-claim-ai",
    version="1.0.0",
    description="AI-Powered Insurance Claim Processing: Multi-Agent System",
    author="Team ClaimFlow AI",
    author_email="team@claimflow.ai",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "openai>=1.50.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "python-multipart>=0.0.9",
        "tiktoken>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "claimflow=app:main",
        ],
    },
)
