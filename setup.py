from distutils.core import setup

setup(
    version='0.0.0',
    scripts=['src/4pointtransform.py', 'src/snow_map.py'],
    packages=['SnowDetectionAndMapping'],
    package_dir={'': 'src'}
)