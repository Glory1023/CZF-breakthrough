'''Setup czf package'''
import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


class CMakeExtension(Extension):
    '''CMake extension that uses 'CMakeLists.txt' files'''
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(_build_ext):
    '''CMake build extensions'''
    def run(self):
        try:
            _ = subprocess.check_output(['cmake', '--version'])
        except OSError as error:
            raise RuntimeError(
                'CMake must be installed to build the following extensions: ' +
                ', '.join(e.name for e in self.extensions)) from error

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = [
            '-DCMAKE_INSTALL_PREFIX=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j']
        if self.parallel:
            build_args.append(str(self.parallel))

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp,
                              env=env)
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', 'install'] + build_args,
            cwd=self.build_temp)


class CMakeTestTarget(CMakeBuild):
    '''CMake test target'''
    def __init__(self, dist):
        super().__init__(dist)
        self.debug = True

    def build_extension(self, ext):
        super().build_extension(ext)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'test'],
                              cwd=self.build_temp)


class CMakeFormatTarget(CMakeBuild):
    '''CMake format target'''
    def build_extension(self, ext):
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', 'clangformat'],
            cwd=self.build_temp)


class CMakeProtoTarget(CMakeBuild):
    '''CMake proto target'''
    def build_extension(self, ext):
        subprocess.check_call(['cmake', '--build', '.', '--target', 'proto'],
                              cwd=self.build_temp)


class CMakeBuildDocs(CMakeBuild):
    '''CMake build docs'''
    def build_extension(self, ext):
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        cfg = 'Debug' if self.debug else 'Release'
        subprocess.check_call(
            ['cmake', ext.sourcedir, '-DCMAKE_BUILD_TYPE=' + cfg],
            cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'sphinx'],
                              cwd=self.build_temp)


with open('README.md') as fd:
    setup(
        name='czf',
        version='1.0',
        author='CGI Lab',
        maintainer='CGI Lab',
        description='czf: CGI Zero Framework',
        long_description=fd.read(),
        long_description_content_type='text/markdown',
        license='BSD-3-Clause',
        ext_modules=[CMakeExtension('czf_env')],
        cmdclass={
            'build_ext': CMakeBuild,
            'build_docs': CMakeBuildDocs,
            'tests': CMakeTestTarget,
            'format': CMakeFormatTarget,
            'proto': CMakeProtoTarget,
        },
        test_suite='tests.test_module',
        tests_require=['pytest'],
        zip_safe=False,
        install_requires=[
            'czf-env @ git+https://github.com/chengscott/czf_env',
            'pyyaml',
            'tensorboard',
            'zmq',
        ],
        packages=find_packages(exclude=['tests']),
        entry_points={
            'console_scripts': [
                'czf-actor = czf.actor:run_main',
                'czf-broker = czf.broker:run_main',
                'czf-game-server = czf.game_server:run_main',
                'czf-model-provider = czf.model_provider:run_main',
                'czf-learner = czf.learner:run_main',
            ],
        },
        classifiers=[
            'License :: OSI Approved :: BSD License',
        ],
    )
