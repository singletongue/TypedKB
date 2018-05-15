#!/usr/bin/env bash

DIR_DATA=./data
DIR_WORK=./work
FILE_ARTICLE=articles.xml.bz2
FILE_META=metadata.xml.bz2
FILE_EXTRACT=extract.json
FILE_EXTRACT_WITH_HYPERLINKS=extract_with_hyperlinks.json
FILE_UDIC_CSV=udic.csv
FILE_UDIC=udic.dic
FILE_CORPUS=corpus.txt
FILE_ENTVEC=entvec.txt
FILE_FEATURES=features.json
FILE_FEATURE2ID=feature2id.json
FILE_REDIRECTS=redirects.json

VECTOR_SIZE=200
N_PROCESS=20

mkdir -p $DIR_WORK

# Extract plain texts from Wikipedia dumps
echo "extracting plain texts from Wikipedia dumps (1/2)"
python wikiextractor/WikiExtractor.py --processes $N_PROCESS --output - --json --no-templates --quiet $DIR_DATA/$FILE_ARTICLE > $DIR_WORK/$FILE_EXTRACT

# Extract plain texts from Wikipedia articles preserving hyperlinks
echo "extracting plain texts from Wikipedia dumps (2/2)"
python wikiextractor/WikiExtractor.py --processes $N_PROCESS --output - --json --links --no-templates --quiet $DIR_DATA/$FILE_ARTICLE > $DIR_WORK/$FILE_EXTRACT_WITH_HYPERLINKS

# Train entity vectors
echo "training entity vectors"
# python scripts/make_udic.py --tokens 1000 --out $DIR_WORK/$FILE_UDIC_CSV
# /usr/local/libexec/mecab/mecab-dict-index -d /usr/local/lib/mecab/dic/ipadic -u $DIR_WORK/$FILE_UDIC -f utf-8 -t utf-8 $DIR_WORK/$FILE_UDIC_CSV
python scripts/tokenize_wikipedia.py --extract $DIR_WORK/$FILE_EXTRACT_WITH_HYPERLINKS --udic $DIR_WORK/$FILE_UDIC --out $DIR_WORK/$FILE_CORPUS
python scripts/train_word2vec.py --corpus $DIR_WORK/$FILE_CORPUS --out $DIR_WORK/$FILE_ENTVEC --size $VECTOR_SIZE --processes $N_PROCESS

# Extract features
echo "extracting features"
python scripts/extract_features.py --article $DIR_DATA/$FILE_ARTICLE --extract $DIR_WORK/$FILE_EXTRACT --entvec $DIR_WORK/$FILE_ENTVEC --out $DIR_WORK/$FILE_FEATURES --feature2id $DIR_WORK/$FILE_FEATURE2ID --redirects $DIR_WORK/$FILE_REDIRECTS
