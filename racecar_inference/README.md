# Inference code

# Useful commands
- restart camera daemon `sudo systemctl restart nvargus-daemon`
- list serial devices `ls /dev/serial/by-path/` or `dmesg | grep tty`
- serial permissions quickfix: `sudo chmod 666 /dev/ttyACM0`
- sort impots and lint (manually bc pre-commit doesn't work out of the box on pytho3.6): `black my-package && isort --profile black my-package`


# Dependencies
Ok so I'm not sure how to handle dependencies on embedded systems running Python? For now a shitty requirements.txt and notes on how to install stuff is what we have.  
Later it would be great to handle dependencies with poetry and/or docker. I probably need to understand more about how operating systems work? Could I get on a call with someone deploying systems on jetson nanos? Or doing research with it?  
[These instructions])(https://github.com/Armandpl/wandb-jetracer/blob/master/JETSON_SETUP.md) are not too bad and then there is a bunch more stuff to fix.  
- I believe we need setuptools-scm==6.0.1, can't remember why
- need MarkupSafe==2.0.1, can't remember why
- need Jetson.GPIO==2.0.21, they say it's more stable than the latest version. The latest version triggers error about permissions.
- pip install protobuf==3.19.6, later version need python3.7

how to use jetson-utils inside venv https://github.com/dusty-nv/jetson-inference/issues/1285

I guess the way to setup this would be:
- create venv
- setup tensorrt
- install pytorch
- install requirements.txt

more instructions here https://github.com/Armandpl/wandb-jetracer/blob/master/JETSON_SETUP.md

# Logger dependencies

I wanted to use the mcap library to log bus traffic but it requires python>=3.7 so we need a second venv.  
It should have pyzmq, mcap, mcap-protobuf-support.
```
sudo apt install python3.8 python3.8-venv
python3.8 -m venv ~/python-envs/rc8
source ~/python-envs/rc8/bin/activate
pip install pip --upgrade
pip install pyzmq mcap mcap-protobuf-support
# inside racecar_inference
pip install -e .
```
