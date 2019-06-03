import setuptools

setuptools.setup(
    name='cifka2019',
    author='Ondřej Cífka',
    description='The code for the paper "Supervised symbolic music style translation using synthetic data"',
    url='https://github.com/cifkao/music-style-translation',
    python_requires='>=3.6',
    install_requires=[
        'museflow @ git+ssh://git@github.com/cifkao/museflow@c14f94ac096743d5a7a83afccfa098853b9c97e9',
        'scipy'
    ],
    packages=setuptools.find_packages(),
)
