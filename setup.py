"""Setup configuration for AI Workflow Engine."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="ai-workflow-engine",
    version="0.1.0",
    author="dickyprase",
    author_email="dickyprase@users.noreply.github.com",
    description="AI-powered workflow automation framework with multi-model routing and intelligent task orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dickyprase/ai-workflow-engine",
    project_urls={
        "Bug Tracker": "https://github.com/dickyprase/ai-workflow-engine/issues",
        "Documentation": "https://github.com/dickyprase/ai-workflow-engine#readme",
        "Source Code": "https://github.com/dickyprase/ai-workflow-engine",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-asyncio>=0.23.0",
            "pytest-cov>=4.1.0",
            "black>=24.1.0",
            "ruff>=0.2.0",
            "mypy>=1.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
    keywords="ai workflow automation llm pipeline orchestration",
)
