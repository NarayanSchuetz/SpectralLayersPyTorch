"""
 Created by Narayan Schuetz at 14/11/2018
 University of Bern
 
 This file is subject to the terms and conditions defined in
 file 'LICENSE.txt', which is part of this source code package.
"""


from setuptools import setup

setup(
    name="spectral",
    version="0.981",
    description="PyTorch NN based trainable spectral linear layers",
    author="Angela Botros & Narayan Schuetz",
    author_email="narayan.schuetz@artorg.unibe.ch",
    license="MIT",
    packages=["spectral"],
    zip_safe=False,
    install_requires=["torch", "numpy"],
)
