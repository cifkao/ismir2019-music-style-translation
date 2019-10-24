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
        'gpu': 'museflow[gpu] @ git+https://github.com/cifkao/museflow@001d7d70416f69565be4874ffe4771a693b9a57b',
        'nogpu': 'museflow[nogpu] @ git+https://github.com/cifkao/museflow@001d7d70416f69565be4874ffe4771a693b9a57b',
    },
    packages=setuptools.find_packages()
)
