#!/usr/bin/env bash

DIR_DATA=./data
TYPE=jawiki
TIMESTAMP=20151123
FILE_ARTICLE=articles.xml.bz2
FILE_META=metadata.xml.bz2

mkdir -p $DIR_DATA

wget "https://archive.org/download/$TYPE-$TIMESTAMP/$TYPE-$TIMESTAMP-pages-articles.xml.bz2" -O "$DIR_DATA/$FILE_ARTICLE"
wget "https://archive.org/download/$TYPE-$TIMESTAMP/$TYPE-$TIMESTAMP-pages-meta-current.xml.bz2" -O "$DIR_DATA/$FILE_META"
