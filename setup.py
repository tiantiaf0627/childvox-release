from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()
    
setup(
    name="childvox",
    version="0.1.0",
    packages=find_packages(),  # auto-detects packages in the folder
    install_requires=required,
    author="Tiantian Feng",
    author_email="tiantiaf@usc.edu",
    description=" a benchmark for understanding and characterizing the diverse acoustic signals through which children communicate, using audio, speech, and large audio-language models.",
    url="https://github.com/tiantiaf0627/childvox-release",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)