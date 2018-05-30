## About

The author's reimplementation of fine-grained named entity classifier proposed in the following paper.

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
```
Entity-based Precision:      88.55%
Entity-based Recall:         88.87%
Entity-based F1 score:       88.40%
Class-based macro Precision: 71.99%
Class-based macro Recall:    57.30%
Class-based macro F1 score:  62.01%
Class-based micro Precision: 92.69%
Class-based micro Recall:    88.13%
Class-based micro F1 score:  90.35%
```

(The final result may slightly vary)
