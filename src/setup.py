from setuptools import setup, find_packages
from glob import glob
from os.path import basename
from os.path import splitext

setup(
    name="Sport radar chart",
    version="0.0.1",
    url='https://github.com/VitalyPavlov/Sport_radar_chart',
    author='Vitaly Pavlov',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "python-dotenv==1.0.1",
        "psycopg2-binary==2.9.9",
        "sqlalchemy==2.0.36",
        "streamlit==1.40.2",
        "pytest==8.3.3",
        "testcontainers==4.8.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "matplotlib==3.9.3",
        "seaborn==0.13.2",
        "statsbombpy",
        "mplsoccer",
        "highlight-text",
        "soccerdata",
        "socceraction",
        "multimethod==1.9",
        "pandera==0.17.2",
        "tqdm==4.67.1",
    ],
)