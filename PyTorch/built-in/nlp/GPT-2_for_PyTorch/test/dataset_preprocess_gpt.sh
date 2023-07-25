# Step 1: Download enwiki
wget `sed '/^enwiki-latest-pages-articles=/!d;s/.*=//' url.ini`
bzip2 -dk enwiki-latest-pages-articles.xml.bz2


# Step 2: Download WikiExtractor
pip3 install wikiextractor
python3 -m wikiextractor.WikiExtractor enwiki-latest-pages-articles.xml --json


# Step3: Concat json
WIKI_DIR=./text
OUTDIR=./data

mkdir -p $OUTDIR
rm $OUTDIR/wiki_all.json
touch $OUTDIR/wiki_all.json

find "$WIKI_DIR" -type f  -print0 |
    while IFS= read -r -d '' line; do
            filename=$(echo "$line" | rev | cut -d'/' -f 1 | rev)
            subfilename=$(echo "$line" | rev | cut -d'/' -f 2 | rev)
            prefix="${subfilename}_${filename}"
            new_name=$(echo "$line")
            echo "Procesing $prefix, $filename, $new_name"
            cat $new_name >> $OUTDIR/wiki_all.json
    done


# Step4: Download Vocab and Do Preprocess
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
VOCAB=./gpt2-vocab.json
MERGE=./gpt2-merges.txt
mv $VOCAB ./data/
mv $MERGE ./data/
python3 tools/preprocess_data.py \
       --input $OUTDIR/wiki_all.json \
       --output-prefix $OUTDIR/my-gpt \
       --vocab $VOCAB \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --split-sentences \
       --workers $(nproc)
