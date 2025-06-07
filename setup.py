from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="magnetic-flux-rope-models", # Name on PyPI
    version="0.1.0",
    author="Jordi Jumilla Lorenz",
    description="Python library for modeling and simulating magnetic flux ropes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jordi-jumilla-lorenz/magnetic-flux-rope-models",
    project_urls={
        "Source": "https://github.com/jordi-jumilla-lorenz/magnetic-flux-rope-models",
        "Tracker": "https://github.com/jordi-jumilla-lorenz/magnetic-flux-rope-models/issues",
    },
    packages=find_packages(include=["magnetic_flux_rope_models", "magnetic_flux_rope_models.*"]),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    license="GPL-3.0-only",
    include_package_data=True,
    install_requires=requirements
)
