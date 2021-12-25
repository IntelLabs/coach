#!/bin/bash

IP_ADRESS=$1
SERVER_END_PATH=$2
CLIENT_END_PATH=$3

scp -r theo.vincent@$1:/home/theovincent/MVA/ObjectRecognition/LearningToAct/coach/$2 /home/developer/LearningToAct/coach/$3