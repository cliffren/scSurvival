#! /usr/bin/env python
 # -*- coding: utf-8 -*_
 # Author: rentao<rentao@amss.ac.cn>
from distutils.core import setup
import setuptools
setup(
    name='scSurvival', # 包的名字
    version='1.0.0',  # 版本号
    description='scSurvival is a new, scalable, and interpretable tool for modeling survival outcomes from single-cell cohort data, with cellular-resolution risk profiling.',  
    author='Tao Ren',
    author_email='renta@ohsu.edu', 
    url='https://github.com/Cliffthinker/scSurvival', 
    packages=setuptools.find_packages(exclude=['*bak', 'example', 'pics', 'libs']),  # 包内不需要引⽤的⽂件夹
    # 依赖包
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'scanpy',
        'scikit-learn',
        'lifelines'
    ],
    zip_safe=False,
    license='GPL-3.0'
)
