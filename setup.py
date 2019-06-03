import setuptools

setuptools.setup(
    name='cifka2019',
    author='Ondřej Cífka',
    description='The code for the paper "Supervised symbolic music style translation using synthetic data"',
    url='https://github.com/cifkao/music-style-translation',
    python_requires='>=3.5',
    install_requires=[
        'museflow @ git+ssh://git@github.com/cifkao/museflow@45d8e8fd7fc03f7a95faba7b0e8ed13a6f0c8db4',
        'scipy'
    ],
    packages=setuptools.find_packages(),
)
