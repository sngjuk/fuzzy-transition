#!/bin/sh

python3 -m virtualenv fuzzy_env
source fuzzy_env/bin/activate
pip3 install -r client_requirements.txt
chmod +x client.py

echo "-=-=-=-= install done -=-=-=--\n\n"

./client.py

