from setuptools import setup
import pandas as pd


ser_ver = pd.read_json("./shafts/shaft_version.json", typ="series", convert_dates=False)
print(ser_ver)
__version__ = f"{ser_ver.ver_milestone}.{ser_ver.ver_major}{ser_ver.ver_remark}"


def readme():
    try:
        with open("./README.md", encoding="utf-8") as f:
            return f.read()
    except:
        return f"SHAFTS package"


setup(
    name="shafts",
    version=__version__,
    description="Simultaneous building Height And FootprinT extraction from Sentinel Imagery",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/LllC-mmd/SHAFTS",
    author=", ".join(
        [
            "Ruidong Li"
            "Dr Ting Sun",
            "Prof Guangheng Ni",
        ]
    ),
    author_email=", ".join(
        [
            "lrd19@mails.tsinghua.edu.cn",
            "ting.sun@reading.ac.uk",
            "ghni@tsinghua.edu.cn",
        ]
    ),
    license="GPL-V3.0",
    packages=["shafts"],
    package_data={
        "shafts": [
            "*.json",
            "utils/*",
            "testCase/*",
            "dl-models/*"
        ]
    },
    # distclass=BinaryDistribution,
    ext_modules=[],
    install_requires=[
        "libarchive",
        "torch",
        "torchvision",
        "tensorflow >= 2.13.0",
        "tensorflow-probability",
        "gdal >= 3.0.0",
        "numpy",
        "matplotlib",
        "scipy", 
        "pandas",
        "albumentations",
        "scikit-learn",
        "scikit-image",
        "xgboost",
        "h5py",
        "shapely",
        "geopandas",
        "rasterio",
        "lmdb",
        "kneed",  # kneed point detection for CCAP algorithms
        "pyarrow",
        "opencv-python",
        "earthengine-api",
        "gcloud"
    ],
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
)