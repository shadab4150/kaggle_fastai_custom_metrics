import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'kaggle_fastai_custom_metrics', 
    version = 'v1.0.2',
    packages = ['kaggle_fastai_custom_metrics'], 
    author = 'shadab_sayeed',                   
    author_email = 'shadabsayeedxxx@gmail.com',
    license = 'MIT', 
    description = 'Custom Metrics for fastai v1 for kaggle competitions', 
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://shadabsayeed.tech/2020-09-08-CustomMetrics/', 
    download_url = 'https://github.com/shadab4150/kaggle_fastai_custom_metrics/archive/v1.0.2.tar.gz',  
    keywords = ['Fastai', 'Competition Metrics', 'Kaggle'],   
    install_requires=[           
          'validators',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
          'torch',
          'fastai',
      ],
    classifiers=[
	"Programming Language :: Python :: 3",
	"License :: OSI Approved :: MIT License",
	"Operating System :: OS Independent",
        ]
)
