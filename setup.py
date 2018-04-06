from setuptools import setup

setup(name='datapipeml',
      version='0.8',
      description='Framework to manipulate dataframes fluidly in a pipeline.',
      url='https://github.com/tiagohcalves/data-pipeline',
      author='Tiago Alves',
      author_email='tiagohcalves@gmail.com',
      license='MIT',
      packages=['datapipeml'],
      install_requires=[
          'pandas',
          'sklearn'
      ],
      zip_safe=False)