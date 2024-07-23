from setuptools import setup, find_packages

setup(
    name="nvtorchcam",
    version="0.1",
    author='Daniel Lichy',
    author_email='dlichy@umd.edu',
    url='https://github.com/NVlabs/nvTorchCam',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
    ],
    extras_require={
        'cubemap': [
            'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git',
        ],
        'examples_tests': [
            'plyfile==1.0',      
            'imageio>=2.31.1', 
            'opencv-python',
            'scipy',  
        ],
        'all': [
            'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git',
            'plyfile==1.0',
            'imageio==2.31.1',
            'opencv-python',
            'scipy',
        ]
    },
    python_requires='>=3.6',
)