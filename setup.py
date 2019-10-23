from setuptools import setup, find_packages

setup(name='clouds',
      version='0.01.1',
      description='Understanding Clouds from Satellite Images',
      url='',
      author='Joseph Chen',
      author_email='',
      license='Apache License Version 2.0, January 2004',
      packages=find_packages(),
      install_requires=[
            "numpy>=1.10.2",
            "scipy",
            "scikit-image",
            "future",
            "pandas",
            "albumentations==0.3.3",
            "torch>=1.2.0",
            "torchvision>=0.4.0",
            "catalyst",
            "lz4",
            "wandb",
            "pytorch_toolbelt",
            "pretrainedmodels",
      ],
      classifiers=[
          'Development Status :: 3 - Alpha',
          # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
          'Intended Audience :: Developers',  # Define that your audience are developers
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',  # Again, pick a license
          'Programming Language :: Python :: 3.6',
      ],
      python_requires='>=3.6',
      keywords=['deep learning', 'image segmentation', 'image classification', 'medical image analysis',
                  'medical image segmentation', 'data augmentation'],
      )
