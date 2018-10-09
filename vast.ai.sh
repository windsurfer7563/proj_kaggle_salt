#!/bin/bash

# before run this script run in console:
#export $KUSERNAME=kaggle_username
#export $KPASSWORD=pass_to_kaggle_accaunt


apt-get --assume-yes  update
apt-get --assume-yes  install vim
apt-get --assume-yes  install libglib2.0-0
apt-get --assume-yes  install libsm6
apt-get --assume-yes  install libxrender1
apt-get --assume-yes  install unzip
apt-get --assume-yes  install zip
apt-get --assume-yes  install mc

pip install --upgrade pip

mkdir -p data/train
mkdir -p data/test

pip install kaggle-cli
#kg download -u ${KUSERNAME} -p ${KPASSWORD} -c tgs-salt-identification-challenge -f train.zip
#kg download -u ${KUSERNAME} -p ${KLASSWORD} -c tgs-salt-identification-challenge -f test.zip



pip install kaggle

unzip train.zip -d data/train
unzip test.zip -d data/test


yes | pip install -r requirements.txt

pip install -U git+https://github.com/albu/albumentations


pip install 'prompt-toolkit==1.0.15'

chmod 777 *.sh

#nohup jupyter notebook --ip=127.0.0.1 --port=8081 --allow-root &