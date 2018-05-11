## About

- 2018年1月に関根先生より依頼があった、2017年11月の Wikipedia 記事に対する固有表現クラスの再タグ付け

## Usage

### 本文抽出

```sh
$ ~/Repos/wikiextractor/WikiExtractor.py --processes 5 --output - --json --no-templates --quiet data/20151123/jawiki-20151123-pages-articles.xml.bz2 > data/20151123/body.json
$ ~/Repos/wikiextractor/WikiExtractor.py --processes 16 --output - --json --no-templates --quiet data/20171103/jawiki-20171103-pages-articles.xml.bz2 > data/20171103/body.json
```

### ベクトル

```sh
$ cp ~/typed_kb/wiki_dumps/ja/20151123/entity_vector.txt data/20151123/
$ cat data/20151123/entity_vector.txt|sed -e 's/^<</\[/g'|sed -e 's/>>\s/\] /g' > data/20151123/entity_vector_square_brackets.txt
```

### 素性抽出

```sh
$ python scripts/preprocess.py --out data/20151123/features.json --id2feature data/20151123/id2feature.json --article data/20151123/jawiki-20151123-pages-articles.xml.bz2 --body data/20151123/body.json --embed data/20151123/entity_vector_square_brackets.txt --ene data/20151123/master20180113.tsv --embed_encoding ISO-8859-1 --n_feature 100000
$ python scripts/preprocess.py --out data/20171103/features.json --id2feature data/20151123/id2feature.json --article data/20171103/jawiki-20171103-pages-articles.xml.bz2 --body data/20171103/body.json --embed data/20151123/entity_vector_square_brackets.txt --embed_encoding ISO-8859-1 --n_feature 100000
```

### 訓練

```sh
$ python scripts/train.py --train data/20151123/features.json --model_dir work/ENEClassifier/f100k --id2ene data/20151123/id2ene.json --gpu 0 --epoch 20 --batch 100 --n_feature 100000
```

### モデルの適用

```sh
$ python scripts/tagging.py --target data/20171103/features.json --model work/ENEClassifier/f100k/model020.npz --id2ene data/20151123/id2ene.json --gpu 1 --batch 100 --n_feature 100000 > data/20171103/enes.json
```


