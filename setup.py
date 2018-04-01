from setuptools import setup

setup(name='datapipe',
      version='0.8',
      description='Framework to manipulate dataframes fluidly in a pipeline.',
      url='https://github.com/tiagohcalves/data-pipeline',
      author='Tiago Alves',
      author_email='tiagohcalves@gmail.com',
      license='MIT',
      packages=['dpipe'],
      install_requires=[
          'pickle',
          'pandas',
          'sklearn'
      ]
      zip_safe=False)