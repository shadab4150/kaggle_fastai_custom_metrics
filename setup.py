from distutils.core import setup
setup(
  name = 'kaggle_fastai_custom_metrics',        
  packages = ['kaggle_fastai_custom_metrics'],  
  version = 'v0.1',      
  license='MIT',        
  description = 'Custom Metrics for fastai v1 for kaggle competitions',  
  author = 'shadab_sayeed',                   
  author_email = 'shadabsayeedxxx@gmail.com',     
  url = 'https://github.com/shadab4150/kaggle_fastai_custom_metrics',  
  download_url = 'https://github.com/shadab4150/kaggle_fastai_custom_metrics/archive/v0.1.tar.gz',  
  keywords = ['Fastai', 'Competition Metrics', 'Kaggle'],   
  install_requires=[           
          'validators',
          'numpy',
          'pandas',
          'sklearn',
          'scipy',
          'torch',
          'fastai',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
