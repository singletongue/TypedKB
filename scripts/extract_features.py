# coding=utf-8
import re
import json
import bz2
import argparse
from collections import defaultdict, Counter
from unicodedata import normalize

import MeCab
from tqdm import tqdm


regex_title = re.compile(r'<title>(.+?)</title>')
regex_id = re.compile(r'<id>(.+?)</id>')
regex_redirect = re.compile(r'<redirect title="(.+?)" />')
regex_heading = re.compile(r'^==(.+?)==$')
regex_category = re.compile(r"\[\[([Cc]ategory|カテゴリ):(.+?)\]\]")


class TextAnalyzer(object):
    def __init__(self):
        self.mt = MeCab.Tagger()

    def tokenize(self, text):
        for line in self.mt.parse(text).split('\n')[:-2]:
            surface = line.split('\t')[0]
            pos_info = line.split('\t')[1].rstrip().split(',')
            token = {
                'surface': surface,
                'pos1': pos_info[0],
                'pos2': pos_info[1]
            }
            yield token

    def extract_ngrams(self, text, n=2, unit='word'):
        assert unit in ('word', 'pos', 'char') and n >= 1
        unigrams = []
        if unit == 'word':
            for token in self.tokenize(text):
                unigrams.append(token['surface'])
        elif unit == 'pos':
            for token in self.tokenize(text):
                unigrams.append(token['pos1'])
        elif unit == 'char':
            unigrams = list(text)

        ngrams = []
        for i in range(0, len(unigrams) - n + 1):
            ngrams.append('_'.join(unigrams[i:i+n]))

        return ngrams

    def extract_nouns(self, text):
        nouns = []
        for token in self.tokenize(text):
            if token['pos1'] == '名詞':
                nouns.append(token['surface'])

        return nouns

    def extract_first_sentence(self, text):
        first_sentence = ''
        for token in self.tokenize(text):
            first_sentence = first_sentence + token['surface']
            if token['pos1'] == '記号' and token['pos2'] == '句点':
                break

        return first_sentence

    def char_type(self, char):
        assert len(char) == 1
        if '\u3040' <= char <= '\u309F':
            return 'ひらがな'
        elif '\u30A0' <= char <= '\u30FF':
            return 'カタカナ'
        elif char.isalpha():
            return '英字'
        else:
            return 'その他'


def main(args):
    analyzer = TextAnalyzer()

    titles = set()
    title2pageid = dict()
    title2headings = defaultdict(list)
    title2categories = defaultdict(list)
    title2first_sentence = defaultdict(str)
    title2embedding = defaultdict(list)
    redirects = dict()

    print('processing the dump file')
    # n_lines = sum(1 for _ in bz2.open(args.article, 'rt'))
    n_lines = 139423487
    with bz2.open(args.article, 'rt') as f:
        title = None
        namespace = None
        for line in tqdm(f, total=n_lines, ncols=60):
            line = line.strip()

            match_title = regex_title.search(line)
            if match_title:
                title = match_title.group(1).replace(' ', '_')
                titles.add(title)

            match_id = regex_id.search(line)
            if match_id:
                pageid = int(match_id.group(1))
                title2pageid.setdefault(title, pageid)

            match_redirect = regex_redirect.search(line)
            if match_redirect:
                src = title
                dst = match_redirect.group(1).replace(' ', '_')
                redirects[src] = dst
                titles.discard(src)

            match_heading = regex_title.search(line)
            if match_heading and namespace == str(0):
                heading = match_heading.group(1)
                title2headings[title].append(heading)

            for match_category in regex_category.finditer(line):
                cat_token = match_category.group(1)
                cat_name = match_category.group(2).replace(' ', '_')
                if '|' in cat_name:
                    cat_name = cat_name[:cat_name.find('|')]
                title2categories[title].append(f'{cat_token}:{cat_name}')

    print('processing the extract file')
    n_articles = sum(1 for _ in tqdm(open(args.extract), unit='lines'))
    with open(args.extract) as f:
        for line in tqdm(f, ncols=60, total=n_articles):
            j = json.loads(line)
            title = j['title'].replace(' ' , '_')
            if len(j['text'].split('\n')) < 3:
                continue
            first_paragraph = j['text'].split('\n')[2].strip()
            first_sentence = analyzer.extract_first_sentence(first_paragraph)
            title2first_sentence[title] = first_sentence

    print('processing the entity vector file')
    with open(args.entvec) as f:
        n_words, n_dim = [int(n) for n in f.readline().rstrip('\n').split()]
        for line in tqdm(f, ncols=60, total=n_words):
            cols = line.rstrip('\n').split()
            if len(cols) != n_dim + 1:
                continue
            if cols[0].startswith('[') and cols[0].endswith(']'):
                title = cols[0][1:-1].replace(' ' , '_')
                embedding = [float(v) for v in cols[1:]]
                title2embedding[title] = embedding


    print('extracting features')
    title2features = defaultdict(list)
    feature_counter = Counter()
    for title in tqdm(titles, ncols=60):
        features = []

        # word [1,2]-grams in the title
        features += [f'TW1_{unigram}' for unigram
                     in analyzer.extract_ngrams(title, n=1, unit='word')]
        features += [f'TW2_{bigram}' for bigram
                     in analyzer.extract_ngrams(title, n=2, unit='word')]

        # POS bigrams in the title
        features += [f'TP2_{pos_bigram}' for pos_bigram
                     in analyzer.extract_ngrams(title, n=2, unit='pos')]

        # last noun in the title
        nouns = analyzer.extract_nouns(title)
        if nouns:
            features += [f'TLN_{nouns[-1]}']

        # last [1,3]-character(s) of the title
        features += [f'TCL1_{title[-1]}']
        if len(title) >= 3:
            features += [f'TCL3_{title[-3:]}']

        # type of the last character of the title
        features += [f'TTL1_{analyzer.char_type(title[-1])}']

        # last noun in the first sentence
        first_sentence = title2first_sentence[title]
        nouns_in_first_sentence = analyzer.extract_nouns(first_sentence)
        if nouns_in_first_sentence:
            features += [f'SLN_{nouns_in_first_sentence[-1]}']
        else:
            features += ['SLN_']

        # headings
        headings = title2headings[title]
        if headings:
            features += [f'H_{heading}' for heading in headings]
        else:
            features += ['H_']

        # last nouns in direct categories
        categories = title2categories[title]
        last_nouns_in_categories = set()
        for cat in categories:
            nouns_in_cat = analyzer.extract_nouns(cat)
            last_nouns_in_categories.update(nouns_in_cat[-1:])

        features += [f'DCLN_{noun}' for noun in last_nouns_in_categories]

        # last nouns in upper categories
        last_nouns_in_upper_categories = set()
        for cat in categories:
            upper_categories = title2categories[cat]
            for ucat in upper_categories:
                nouns_in_cat = analyzer.extract_nouns(ucat)
                last_nouns_in_upper_categories.update(nouns_in_cat[-1:])

        features += [f'UCLN_{noun}' for noun in last_nouns_in_upper_categories]

        features = list(set(features))
        title2features[title] = features
        feature_counter.update(features)

    id2feature = [feature for feature, _ in
                  feature_counter.most_common()]
    feature2id = {feature: i for i, feature in enumerate(id2feature)}

    print('writing out extracted features')
    with open(args.out, 'w') as fo:
        for title in tqdm(titles, ncols=60):
            item = {
                'title': title,
                'feature_ids': [feature2id[f] for f in title2features[title]],
                'embedding': title2embedding[title]
            }
            print(json.dumps(item, ensure_ascii=False), file=fo)

    json.dump(feature2id, open(args.feature2id, 'w'), ensure_ascii=False)
    json.dump(redirects, open(args.redirects, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--article', type=str, required=True)
    parser.add_argument('--extract', type=str, required=True)
    parser.add_argument('--entvec', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--feature2id', type=str, required=True)
    parser.add_argument('--redirects', type=str, required=True)

    args = parser.parse_args()
    main(args)
