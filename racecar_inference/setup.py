from setuptools import find_packages, setup

setup(
    name="racecar_inference",
    version="0.1",
    packages=find_packages(),
    description="skrt skrrrt",
    author="Armand",
    author_email="adpl33@gmail.com",
    url="https://github.com/armandpl/skyline",
    install_requires=[
        "adafruit-circuitpython-servokit"  # for jetracer
        # TODO delete setup.py and use poetry to handle dependencies
    ],
)
