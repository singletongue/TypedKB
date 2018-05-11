# coding=utf-8

import os
import json
import argparse
import logging
from collections import Counter
import numpy as np
import chainer
from chainer import cuda
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from tqdm import tqdm, trange


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                    datefmt='%m/%d %H:%M:%S')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', type=str, required=True,
                    help='training file (.json)')
parser.add_argument('--model_dir', type=str, required=True,
                    help='directory to save model files')
parser.add_argument('--id2ene', type=str, required=True,
                    help='path to save id2ene file')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', type=int, default=10,
                    help='number of epochs for training the model')
parser.add_argument('--batch', type=int, default=10,
                    help='minibatch size')
parser.add_argument('--n_feature', type=int, default=10000,
                    help='number of features (default=10000)')
parser.add_argument('--ndim_embedding', type=int, default=200,
                    help='number of dimensions in embedding')
args = parser.parse_args()

FNAME_TRAIN = args.train
DIR_MODEL = args.model_dir
FNAME_ID2ENE = args.id2ene
GPU = args.gpu
N_EPOCH = args.epoch
BATCH_SIZE = args.batch
N_FEATURE = args.n_feature
NDIM_EMBEDDING = args.ndim_embedding

if GPU >= 0:
    cuda.get_device_from_id(GPU).use()
    import cupy as cp
    xp = cp
else:
    xp = np

np.random.seed(912)
xp.random.seed(912)

logging.info('counting lines of training file')
nline = sum(1 for _ in open(FNAME_TRAIN))

logging.info('preprocessing training file')
x_featuress_train = []
x_embeddings_train = []
y_ene_namess_train = []
ene_counter = Counter()

with open(FNAME_TRAIN) as f:
    for line in tqdm(f, total=nline, dynamic_ncols=True):
        item = json.loads(line)
        if 'enes' not in item:
            continue
        x_featuress_train.append(item.get('feature_ids'))
        x_embeddings_train.append(item.get('embedding', None))
        y_ene_namess_train.append(item.get('enes'))
        ene_counter.update(item.get('enes'))

n_train = len(x_featuress_train)

id2ene = [ene for ene, _ in ene_counter.most_common()]
ene2id = {ene: i for i, ene in enumerate(id2ene)}
y_eness_train = [[ene2id[ene] for ene in enes] for enes in y_ene_namess_train]

n_enes = len(ene2id)

json.dump(id2ene, open(FNAME_ID2ENE, 'w'))

class ENEClassifier(chainer.Chain):
    def __init__(self, ndim_in, ndim_mid, ndim_out):
        super(ENEClassifier, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_size=ndim_in, out_size=ndim_mid)
            self.l2 = L.Linear(in_size=ndim_mid, out_size=ndim_out)
        
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x):
        h = F.tanh(self.l1(x))
        y_pred = self.l2(h)
        return y_pred

model = ENEClassifier(ndim_in=N_FEATURE+NDIM_EMBEDDING,
                      ndim_mid=200,
                      ndim_out=n_enes)

optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

if GPU >= 0:
    model.to_gpu(device=GPU)

logging.info('training the model')
for epoch in trange(N_EPOCH, dynamic_ncols=True):
    total_loss = 0
    perm = np.random.permutation(n_train)
    for i in trange(0, n_train, BATCH_SIZE, dynamic_ncols=True):
        batch_indices = perm[i:i+BATCH_SIZE]
        b = batch_indices.size

        x_featuress_batch = [x_featuress_train[i] for i in batch_indices]
        x_embeddings_batch = [x_embeddings_train[i] for i in batch_indices]
        y_eness_batch = [y_eness_train[i] for i in batch_indices]
        
        def x2tensor(x_featuress, x_embeddings):
            assert len(x_featuress) == len(x_embeddings)
            b = len(x_featuress)
            xf = xp.zeros(shape=(b, N_FEATURE), dtype=xp.float32)
            for i, features in enumerate(x_featuress):
                for feature in features:
                    if feature < N_FEATURE:
                        xf[i][feature] += 1.0
            xe = xp.zeros(shape=(b, NDIM_EMBEDDING), dtype=xp.float32)
            for i, embedding in enumerate(x_embeddings):
                if embedding:
                    xe[i,:] = xp.array(embedding, dtype=xp.float32)
            return xp.concatenate((xf, xe), axis=1)

        def y2tensor(y_eness):
            b = len(y_eness)
            c = n_enes
            y = xp.zeros(shape=(b, c), dtype=xp.int32)
            for i, enes in enumerate(y_eness):
                for ene in enes:
                    y[i][ene] = 1.0
            return y

        x = x2tensor(x_featuress_batch, x_embeddings_batch)
        y = y2tensor(y_eness_batch)

        y_pred = model(x)

        loss = F.sigmoid_cross_entropy(y_pred, y)
        total_loss += float(F.sum(loss).data)

        model.cleargrads()
        loss.backward()
        optimizer.update()

    print('loss = {:.4f}'.format(total_loss))
    fname_model = 'model{:03d}.npz'.format(epoch + 1)
    save_path = os.path.join(DIR_MODEL, fname_model)
    chainer.serializers.save_npz(save_path, model)
