import re
import argparse
import logging

from gensim.models.word2vec import LineSentence, Word2Vec


logging.basicConfig(level=logging.INFO)


def main(args):
    logging.info('building entity vectors')
    model = Word2Vec(sentences=LineSentence(args.corpus),
                     size=args.size, workers=args.processes, sg=1)

    logging.info('saving entity vectors')
    model.wv.save_word2vec_format(args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--size', type=int, default=200)
    parser.add_argument('--processes', type=int, default=1)
    args = parser.parse_args()

    main(args)
