from setuptools import setup, find_packages

setup(
    name="stk-apex",
    version="2.0.0",
    packages=find_packages(),
    package_data={"stk_apex": ["tokenizer/*.json"]},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    extras_require={
        "rag": ["rank-bm25"],
        "assistant": ["httpx", "beautifulsoup4"],
        "dev": ["pytest"],
    },
)
