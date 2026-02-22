from setuptools import setup, find_packages

setup(
    name="agentic_mmm",
    version="2.0.0",
    # Map the "agentic_mmm" package to the current directory
    package_dir={"agentic_mmm": "."},
    # Explicitly list the packages so setuptools knows where they are
    packages=[
        "agentic_mmm",
        "agentic_mmm.agent",
        "agentic_mmm.core",
        "agentic_mmm.tools",
        "agentic_mmm.workflows",
    ],
    install_requires=[
        "langchain-core",
        "langchain-databricks",
        "langgraph",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "pymc",
        "pydantic",
        "rich"
    ],
)
