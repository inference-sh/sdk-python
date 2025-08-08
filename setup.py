from setuptools import setup, find_packages

setup(
    name="inferencesh",
    version="0.1.2",
    description="inference.sh Python SDK",
    author="Inference Shell Inc.",
    author_email="hello@inference.sh",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic>=2.0.0",
        "tqdm>=4.67.0",
        "requests>=2.31.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 