from setuptools import setup, find_packages

setup(
    name='hybrideb',
    description="hybrid real/Fourier-space E/B-mode estimators for cosmic shear",
    author="Matthew R. Becker",
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
