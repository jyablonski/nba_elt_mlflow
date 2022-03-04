#!/bin/bash

set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update

apt-get -y upgrade

apt-get -y install --no-install-recommends gcc
pip install --upgrade pip
pip install pipenv

apt-get clean
rm -rf /var/lib/apt/lists/*