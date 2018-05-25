## About

An implementation of fine-grained named entity classifier proposed in the following paper.

- Masatoshi Suzuki, Koji Matsuda, Satoshi Sekine, Naoaki Okazaki and Kentaro Inui. *A Joint Neural Model for Fine-Grained Named Entity Classification of Wikipedia Articles.* IEICE Transactions on Information and Systems, Special Section on Semantic Web and Linked Data, Vol. E101-D, No.1, pp.73-81, 2018. [Link](https://www.jstage.jst.go.jp/article/transinf/E101.D/1/E101.D_2017SWP0005/_article/-char/ja/)

## Usage

### Download Wikipedia dump file

```sh
$ chmod +x download.sh; ./download.sh
```

### Preprocess downloaded files (including feature extraction)

```sh
$ chmod +x preprocess.sh; ./preprocess.sh
```

### Train and test classifiers (cross validation)

```sh
$ chmod +x cross_validation.sh; ./cross_validation.sh
```
