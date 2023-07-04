#!/usr/bin/env bash

set -e

echo "Downloading cmudict-0.7b ..."
cmudict=`sed '/^cmudict=/!d;s/.*=//' url.ini`
wget ${cmudict} -qO cmudict/cmudict-0.7b
