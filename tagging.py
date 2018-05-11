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
parser.add_argument('--target', type=str, required=True,
                    help='target file (.json)')
parser.add_argument('--model', type=str, required=True,
                    help='model file (.npz)')
parser.add_argument('--id2ene', type=str, required=True,
                    help='id2ene file (.json)')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--batch', type=int, default=10,
                    help='minibatch size')
parser.add_argument('--n_feature', type=int, default=10000,
                    help='number of features (default=10000)')
parser.add_argument('--ndim_embedding', type=int, default=200,
                    help='number of dimensions in embedding')
args = parser.parse_args()

FNAME_TARGET = args.target
FNAME_MODEL = args.model
FNAME_ID2ENE = args.id2ene
GPU = args.gpu
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
nline = sum(1 for _ in open(FNAME_TARGET))

logging.info('preprocessing training file')
x_pageids_target = []
x_titles_target = []
x_featuress_target = []
x_embeddings_target = []
y_ene_namess_target = []

with open(FNAME_TARGET) as f:
    for line in tqdm(f, total=nline):
        item = json.loads(line)
        x_pageids_target.append(item.get('pageid'))
        x_titles_target.append(item.get('title'))
        x_featuress_target.append(item.get('feature_ids'))
        x_embeddings_target.append(item.get('embedding', None))

n_target = len(x_featuress_target)

id2ene = json.load(open(FNAME_ID2ENE))
n_enes = len(id2ene)

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
chainer.serializers.load_npz(FNAME_MODEL, model)

if GPU >= 0:
    model.to_gpu(device=GPU)

logging.info('tagging the target')
for i in trange(0, n_target, BATCH_SIZE):
    x_pageids_batch = x_pageids_target[i:i+BATCH_SIZE]
    x_titles_batch = x_titles_target[i:i+BATCH_SIZE]
    x_featuress_batch = x_featuress_target[i:i+BATCH_SIZE]
    x_embeddings_batch = x_embeddings_target[i:i+BATCH_SIZE]
    
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

    x = x2tensor(x_featuress_batch, x_embeddings_batch)
    with chainer.using_config('train', False):
        y = F.sigmoid(model(x))
    y_prob = chainer.cuda.to_cpu(y.data)

    y_pred = (y_prob >= 0.5) + (y_prob == np.max(y_prob, axis=1).reshape(-1,1))
    assert len(x_titles_batch) == y_pred.shape[0]
    for i in range(len(x_titles_batch)):
        yi_indices = np.where(y_pred[i]==True)[0].tolist()
        yi_probs = y_prob[i][yi_indices]
        yi_enes = [{'ENE': id2ene[j], 'prob': float(prob)} for j, prob in zip(yi_indices, yi_probs)]
        yi_json = {'pageid': x_pageids_batch[i], 'title': x_titles_batch[i], 'ENEs': yi_enes}
        print(json.dumps(yi_json, ensure_ascii=False))

