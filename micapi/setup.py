import setuptools
import os

version_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
version_file = os.path.join(version_dir, 'micapi', '__init__.py')
exec(open(version_file, 'r').read())
print(version_file)


setuptools.setup(name='micapi',
                 version=__version__,
                 description='micapi',
                 author='Dorel Masasa',
                 url='http://eltvxgitlab01/6263/Tests/epl-creator',
                 classifiers=["Programming Language :: Python :: 3"])

