#!/usr/bin/env bash

set -e

scripts/download_cmudict.sh

DATA_DIR="LJSpeech-1.1"
LJS_ARCH="LJSpeech-1.1.tar.bz2"
speech=`sed '/^speech=/!d;s/.*=//' url.ini`
LJS_URL="${speech}${LJS_ARCH}"

if [ ! -d ${DATA_DIR} ]; then
  echo "Downloading ${LJS_ARCH} ..."
  wget -q ${LJS_URL}
  echo "Extracting ${LJS_ARCH} ..."
  tar jxvf ${LJS_ARCH}
  rm -f ${LJS_ARCH}
fi
