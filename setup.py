from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='datapipeml',
      version='0.8',
      description='Framework to manipulate dataframes fluidly in a pipeline.',
      long_description=readme(),
      keywords='pandas pipeline machine learning dataset etl',
      url='https://github.com/tiagohcalves/data-pipeline',
      author='Tiago Alves',
      author_email='tiagohcalves@gmail.com',
      license='MIT',
      packages=['datapipeml'],
      install_requires=[
          'pandas',
          'sklearn'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'])
