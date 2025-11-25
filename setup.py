from setuptools import setup, find_packages

setup(
    name='sentiment_analyzer',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={"":"src"},    
    install_requires=[
        "numpy", "pandas", "matplotlib", "scikit-learn", "mlflow", "click"
    ],
    entry_points={
        'console_scripts': [
            'predict=sentiment_analyzer.predict:main',
            'promote=sentiment_analyzer.promote:main'
        ],
    },
)