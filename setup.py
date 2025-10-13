from setuptools import setup, find_packages

setup(
    name="rl-sutton-barto",
    version="0.1.0",
    author="rgb",
    description="Implementations of RL algorithms from Sutton & Barto's book",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    python_requires=">=3.8",
)

