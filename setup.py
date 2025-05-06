from setuptools import setup, find_packages

setup(
    name="speechBCI",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "transformers",
        "tensorboard",
        "tqdm",
        "matplotlib",
        "umap-learn",
        "ray[tune]",
    ],
    python_requires=">=3.8",
) 