#!/bin/bash

# Update apt-get
sudo apt-get update upgrade

# Get python package installer and the webserver nginx
sudo apt-get install python3-pip nginx

#  install required packages
pip3 install -r requirements.txt
