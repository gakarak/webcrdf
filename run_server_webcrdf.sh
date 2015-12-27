#!/bin/bash

source ~/.profile

##source ~/dev/bin/set-ocv-cuda-jdk-qt4.sh
source ~/dev/bin/set-ocv-qt4.sh

source $HOME/venv/venv-pycharm/bin/activate

cd webcrdf

##python manage.py runserver 192.168.0.10:8001
##python manage.py runserver 192.168.219.12:8001
##python manage.py runserver 80.94.162.115:8080
python manage.py runserver 127.0.0.1:8080 --insecure

