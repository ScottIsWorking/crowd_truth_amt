 # drawing from here these two resources to design this
 #  https://github.com/navdeep-G/setup.py
 # https://docs.python-guide.org/writing/structure/

from setuptools import setup

setup(name='crowdtruth_amt',
      version='0.1',
      description='metrics for use with multilabel turk annotation tasks',
      author='Scott Cambo',
      author_email='scott@avalancheinsights.com',
      packages=['crowd_truth_amt'],
      zip_safe=False,
      install_requires=[efficient-apriori==1.1.1,
      json5==0.9.5,
      numpy==1.19.0,
      pandas==1.0.5,
      pandocfilters==1.4.2,
      qgrid==1.3.1,
      regex==2020.6.8,
      scikit-learn==0.23.1,
      scipy==1.5.0,
      tqdm==4.46.1,
      urllib3==1.25.9])
