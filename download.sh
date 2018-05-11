#!/usr/bin/env bash

DATA_DIR=./data

mkdir $DATA_DIR

wget https://archive.org/download/jawiki-20151123/jawiki-20151123-pages-articles.xml.bz2 -O $DATA_DIR
wget https://archive.org/download/jawiki-20151123/jawiki-20151123-pages-meta-current.xml.bz2 -O $DATA_DIR
