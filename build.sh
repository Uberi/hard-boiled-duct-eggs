#!/usr/bin/env bash

sudo apt-get install libboost-all-dev
sudo apt-get install python
pip install wheel
pip install face_recognition

# install OpenCV
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install python3.5-dev
git clone --depth=1 https://github.com/Itseez/opencv.git
cd opencv/ && git checkout
cd .. && mkdir build/ && cd build/
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ../opencv/
make
sudo make install
