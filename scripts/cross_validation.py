import json
import random
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, reporter
from chainer.links import Classifier
from chainer.dataset import concat_examples
from chainer.datasets import get_cross_validation_datasets
from chainer.backends import cuda
from chainer.iterators import SerialIterator
from chainer.training import StandardUpdater, Trainer, extensions


class ENEClassifier(chainer.Chain):
    def __init__(self, in_size, hidden_size, out_size):
        super(ENEClassifier, self).__init__()
        self.loss = None
        with self.init_scope():
            self.encoder = L.Linear(in_size, hidden_size)
            self.decoder = L.Linear(hidden_size, out_size)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x, t):
        h = F.tanh(self.encoder(x))
        y_logit = self.decoder(h)
        self.loss = F.sigmoid_cross_entropy(y_logit, t)
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def predict(self, x):
        h = F.tanh(self.encoder(x))
        y_logit = self.decoder(h)
        y = F.sigmoid(y_logit)
        return y


def load_dataset(fpath_dataset, fpath_features, fpath_redirects):
    redirects = json.load(open(fpath_redirects))

    dataset = []
    with open(fpath_dataset) as f:
        for line in f:
            article = json.loads(line)
            dataset.append(article)

    features = dict()
    with open(fpath_features) as f:
        for line in f:
            article = json.loads(line)
            features[article['title']] = {
                'feature_ids': article['feature_ids'],
                'embedding': article['embedding']
            }

    ene_counter = Counter()
    for article in dataset:
        ene_counter.update(article['enes'])

    id2ene = [ene for ene, _ in ene_counter.most_common()]
    ene2id = {ene: idx for idx, ene in enumerate(id2ene)}

    processed_dataset = []
    for article in dataset:
        title = article['title'].replace(' ', '_')
        if title in redirects:
            title = redirects[title]

        item = {
            'title': title,
            'feature_ids': features[title]['feature_ids'],
            'embedding': features[title]['embedding'],
            'ene_ids': [ene2id[ene] for ene in article['enes']]
        }
        processed_dataset.append(item)

    return processed_dataset, id2ene


def main(args):
    random.seed(0)
    np.random.seed(0)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        cuda.cupy.random.seed(0)

    dataset, id2ene = load_dataset(args.dataset, args.features, args.redirects)
    print(f'# of examples in dataset: {len(dataset)}')

    def batch2tensors(batch, device):
        xp = cuda.cupy if device >= 0 else np

        xf = xp.zeros((len(batch), args.n_feature), dtype='f')
        xe = xp.zeros((len(batch), args.embed_size), dtype='f')
        t = xp.zeros((len(batch), len(id2ene)), dtype='i')

        for i, item in enumerate(batch):
            for feature_id in item['feature_ids']:
                if feature_id < args.n_feature:
                    xf[i, feature_id] = 1.0

            if item['embedding']:
                xe[i] = xp.array(item['embedding'], dtype='f')

            for ene_id in item['ene_ids']:
                t[i, ene_id] = 1

        x = xp.concatenate((xf, xe), axis=1)

        return x, t

    cv_datasets = get_cross_validation_datasets(dataset, args.cv)
    ys = []
    ts = []
    for split_idx, cv_dataset in enumerate(cv_datasets):
        print(f'cross validation ({split_idx + 1}/{len(cv_datasets)})')
        train, test = cv_dataset
        train_iter = SerialIterator(train, batch_size=args.batch)
        test_iter = SerialIterator(test, batch_size=args.batch,
                                   repeat=False, shuffle=False)

        model = ENEClassifier(in_size=args.n_feature + args.embed_size,
                              hidden_size=args.hidden_size,
                              out_size=len(id2ene))

        if args.gpu >= 0:
            model.to_gpu(args.gpu)

        optimizer = optimizers.Adam()
        optimizer.setup(model)
        updater = StandardUpdater(train_iter, optimizer,
                                  converter=batch2tensors, device=args.gpu)

        trainer = Trainer(updater, (args.epoch, 'epoch'), out=args.out_dir)
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.snapshot_object(
            model, filename='epoch_{.updater.epoch}.model'))
        trainer.extend(extensions.Evaluator(
            test_iter, model, converter=batch2tensors, device=args.gpu))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar(update_interval=1))

        trainer.run()

        test_iter.reset()
        for batch in test_iter:
            x, t = batch2tensors(batch, device=args.gpu)
            with chainer.using_config('train', False):
                y = model.predict(x)

            ys.append(y)
            ts.append(t)

    y_all = F.concat(ys, axis=0)
    t_all = F.concat(ts, axis=0)

    prediction_matrix = (y_all.data >= 0.5).astype('f')
    reference_matrix = (t_all.data == 1).astype('f')
    accuracy_matrix = prediction_matrix * reference_matrix

    eb_pred = prediction_matrix.sum(axis=1)  # entity-based num. of predicted classes
    eb_ref = reference_matrix.sum(axis=1)  # entity-based num. of reference classes
    eb_acc = accuracy_matrix.sum(axis=1)  # entity-based num. of accurate classes

    eb_nopred = (eb_pred==0.).astype('f')  # for avoiding zero-division
    eb_precision = (eb_acc / (eb_pred + eb_nopred)).mean()
    eb_recall = (eb_acc / eb_ref).mean()
    eb_f1 = (2 * eb_acc / (eb_pred + eb_ref)).mean()

    cb_pred = prediction_matrix.sum(axis=0)  # class-based num. of predicted examples
    cb_ref = reference_matrix.sum(axis=0)  # class-based num. of reference examples
    cb_acc = accuracy_matrix.sum(axis=0)  # class-based num. of accurate examples

    cb_nopred = (cb_pred==0.).astype('f')  # for avoiding zero-division
    cb_macro_precision = (cb_acc / (cb_pred + cb_nopred)).mean()
    cb_macro_recall = (cb_acc / cb_ref).mean()
    cb_macro_f1 = (2 * cb_acc / (cb_pred + cb_ref)).mean()

    cb_micro_precision = cb_acc.sum() / cb_pred.sum()
    cb_micro_recall = cb_acc.sum() / cb_ref.sum()
    cb_micro_f1 = (2 * cb_acc.sum()) / (cb_pred.sum() + cb_ref.sum())

    print(f'Entity-based Precision:      {float(eb_precision):.2%}')
    print(f'Entity-based Recall:         {float(eb_recall):.2%}')
    print(f'Entity-based F1 score:       {float(eb_f1):.2%}')

    print(f'Class-based macro Precision: {float(cb_macro_precision):.2%}')
    print(f'Class-based macro Recall:    {float(cb_macro_recall):.2%}')
    print(f'Class-based macro F1 score:  {float(cb_macro_f1):.2%}')

    print(f'Class-based micro Precision: {float(cb_micro_precision):.2%}')
    print(f'Class-based micro Recall:    {float(cb_micro_recall):.2%}')
    print(f'Class-based micro F1 score:  {float(cb_micro_f1):.2%}')

    with open(Path(args.out_dir) / 'classification_result.json', 'w') as fo:
        for i, item in enumerate(dataset):
            title = item['title']
            predicted_classes = [id2ene[j]
                for j, v in enumerate(prediction_matrix[i]) if v == 1.0]
            reference_classes = [id2ene[j]
                for j, v in enumerate(reference_matrix[i]) if v == 1.0]
            out = {
                'title': title,
                'prediction': predicted_classes,
                'reference': reference_classes
            }
            print(json.dumps(out, ensure_ascii=False), file=fo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--redirects', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--cv', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--n_feature', type=int, default=10000)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    main(args)
