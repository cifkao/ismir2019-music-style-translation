import setuptools

setuptools.setup(
    name='ismir2019_cifka',
    author='Ondřej Cífka',
    description='The code for the paper "Supervised symbolic music style translation using synthetic data"',
    url='https://github.com/cifkao/music-style-translation',
    python_requires='>=3.6',
    install_requires=[
        'scipy'
    ],
    extras_require={
        'gpu': 'museflow[gpu] @ git+ssh://git@github.com/cifkao/museflow@d9607bdb00a465f338c12b24a3bb3c1e0742650d',
        'nogpu': 'museflow[nogpu] @ git+ssh://git@github.com/cifkao/museflow@d9607bdb00a465f338c12b24a3bb3c1e0742650d',
    },
    packages=setuptools.find_packages()
)
