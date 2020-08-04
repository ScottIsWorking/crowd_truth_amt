 # drawing from here these two resources to design this
 #  https://github.com/navdeep-G/setup.py
 # https://docs.python-guide.org/writing/structure/

from setuptools import setup

with open('requirements.txt') as fIn:
    requirements=fIn.read().splitlines()

setup(name='crowdtruth_amt',
      version='0.1',
      description='metrics for use with multilabel turk annotation tasks',
      author='Scott Cambo',
      author_email='scott@avalancheinsights.com',
      packages=['crowd_truth_amt'],
      zip_safe=False,
      install_requires=[efficient-apriori,
      json5,
      numpy,
      pandas,
      pandocfilters,
      qgrid,
      regex,
      scikit-learn,
      scipy,
      tqdm,
      urllib3])
