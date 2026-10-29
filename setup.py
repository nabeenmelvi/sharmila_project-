from setuptools import setup, find_packages

setup(
    name="verimed-rag",
    version="0.1.0",
    description="Biomedical Retrieval-Augmented Generation benchmark package",
    author="Sharmila",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "rank-bm25>=0.2.2",
        "biopython>=1.81",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
    ],
)
