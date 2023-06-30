#!/bin/bash

# Download directories vars
root_dl="k400"
root_dl_targz="k400_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# Download validation tars, will resume
curr_dl=${root_dl_targz}/val
url=`sed '/^val_url=/!d;s/.*=//' url.ini`
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i $url -P $curr_dl

# Download annotations csv files
curr_dl=${root_dl}/annotations
url_v=`sed '/^anno_url=/!d;s/.*=//' url.ini`
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c $url_v -P $curr_dl

# Download readme
url=`sed '/^readme_url=/!d;s/.*=//' url.ini`
wget -c $url -P $root_dl

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k400_extractor.sh"
