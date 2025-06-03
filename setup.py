from setuptools import setup, find_packages

setup(
    name="magnetic-flux-rope-models", # Name on PyPI
    version="0.1.0",
    author="Jordi Jumilla Lorenz",
    description="Python library for modeling and simulating magnetic flux ropes.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/magnetic-flux-rope-models",
    project_urls={
        "Source": "https://github.com/yourusername/magnetic-flux-rope-models",
        "Tracker": "https://github.com/yourusername/magnetic-flux-rope-models/issues",
    },
    packages=find_packages(include=["magnetic_flux_rope_models", "magnetic_flux_rope_models.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    license="GPL-3.0-only",
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
    ],
)
