mkdir -p ./data/
ZIP_FILE=./data/celeba.zip
mv ./celeba.zip $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE