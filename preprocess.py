# coding=utf-8
import re
import json
import bz2
import argparse
import MeCab
from collections import defaultdict, Counter
from unicodedata import normalize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, required=True,
                    help='output file (.json)')
parser.add_argument('--id2feature', type=str, required=True,
                    help='id2feature file to load/save (.json)')
parser.add_argument('--article', type=str, required=True,
                    help='Wikipedia dump file (.xml.bz2)')
parser.add_argument('--body', type=str, required=True,
                    help='output of Wikiextractor.py (.json)')
parser.add_argument('--embed', type=str, required=True,
                    help='word2vec output file (.txt)')
parser.add_argument('--ene', type=str, default='',
                    help='ene tagged file (.tsv) [title TAB ene1;ene2;...]')
parser.add_argument('--embed_encoding', type=str, default='utf-8',
                    help='word2vec output file encoding format')
parser.add_argument('--n_feature', type=int, default=10000,
                    help='number of features (default=10000)')


args = parser.parse_args()

FNAME_OUT = args.out
FNAME_ID2FEATURE = args.id2feature
FNAME_ARTICLE = args.article
FNAME_BODY = args.body
FNAME_EMBED = args.embed
FNAME_ENE = args.ene
EMBED_ENCODING = args.embed_encoding
N_FEATURE = args.n_feature

mt = MeCab.Tagger()

def ngrams(text, n=2, unit='word'):
    assert unit in ('word', 'pos', 'char') and n >= 1
    unigrams = []
    if unit == 'word':
        for line in mt.parse(text).split('\n')[:-2]:
            word = line.split('\t')[0]
            unigrams.append(word)
    elif unit == 'pos':
        for line in mt.parse(text).split('\n')[:-2]:
            pos = line.split('\t')[1].split(',')[0]
            unigrams.append(pos)
    elif unit == 'char':
        unigrams = list(text)
    
    ngrams = []
    for i in range(0, len(unigrams) - n + 1):
        ngrams.append('_'.join(unigrams[i:i+n]))
    
    return ngrams

def nouns(text):
    nouns = []
    for line in mt.parse(text).split('\n')[:-2]:
        word = line.split('\t')[0]
        pos = line.split('\t')[1].split(',')[0]
        if pos == '名詞':
            nouns.append(word)
    
    return nouns

def char_type(char):
    assert len(char) == 1
    if '\u3040' <= char <= '\u309F':
        return 'ひらがな'
    elif '\u30A0' <= char <= '\u30FF':
        return 'カタカナ'
    elif char.isalpha():
        return '英字'
    else:
        return 'その他'

regex_title = re.compile(r'<title>(.+?)</title>')
regex_ns = re.compile(r'<ns>(.+?)</ns>')
regex_id = re.compile(r'<id>(.+?)</id>')
regex_redirect = re.compile(r'<redirect title="(.+?)" />')
regex_heading = re.compile(r'^==(.+?)==$')
regex_category =re.compile(r"\[\[(Category|category|カテゴリ):(.+?)(\|.+?)?\]\]")
regex_np = re.compile(r'([^ぁ-ん]+?)(の(一種|総称|愛称|呼称|名称|名前|名|1つ|ひとつ|一つ|一|こと))?(である)?。')

def extract_first_sentence(text):
    first_sentence = ''
    for line in mt.parse(text).split('\n')[:-2]:
        word = line.split('\t')[0]
        pos1, pos2 = line.split('\t')[1].split(',')[0:2]
        first_sentence = first_sentence + word
        if pos1 == '記号' and pos2 == '句点':
            break
    return first_sentence


titles = set()
pageids = dict()
headingss = defaultdict(list)
categoriess = defaultdict(list)
first_sentences = dict()
embeddings = dict()
eness = defaultdict(list)

# nline_fa = sum(1 for _ in tqdm(bz2.open(FNAME_ARTICLE, 'rt')))
nline_fa = 139423487
with bz2.open(FNAME_ARTICLE, 'rt') as fa:
    title = ''
    ns = ''
    for line in tqdm(fa, total=nline_fa):
        line = line.strip()

        match_title = regex_title.search(line)
        if match_title:
            title = match_title.group(1).replace(' ' , '_')
        
        match_ns = regex_ns.search(line)
        if match_ns:
            ns = match_ns.group(1)
            if ns == str(0):
                titles.add(title)

        match_id = regex_id.search(line)
        if match_id:
            pageid = int(match_id.group(1))
            pageids.setdefault(title, pageid)

        match_redirect = regex_redirect.search(line)
        if match_redirect:
            titles.discard(title)
        
        match_heading = regex_title.search(line)
        if match_heading and ns == str(0):
            heading = match_heading.group(1)
            headingss[title].append(heading)
        
        for match_category in regex_category.finditer(line):
            if ns == str(0) or ns == str(14):
                category = match_category.group(2)
                categoriess[title].append(category)

nline_fb = sum(1 for _ in tqdm(open(FNAME_BODY)))
with open(FNAME_BODY) as fb:
    for line in tqdm(fb, total=nline_fb):
        j = json.loads(line)
        title = j['title'].replace(' ' , '_')
        if len(j['text'].split('\n')) < 3:
            continue
        first_paragraph = j['text'].split('\n')[2].strip()
        first_sentence = extract_first_sentence(first_paragraph)
        first_sentences[title] = first_sentence

n_words, n_dim = None, None
with open(FNAME_EMBED, encoding=EMBED_ENCODING) as fe:
    n_words, n_dim = [int(n) for n in fe.readline().rstrip('\n').split()]
    for line in tqdm(fe, total=n_words):
        cols = line.rstrip('\n').split()
        if len(cols) != n_dim + 1:
            continue
        if cols[0].startswith('[') and cols[0].endswith(']'):
            title = cols[0].lstrip('[').rstrip(']').replace(' ' , '_')
            embedding = [float(v) for v in cols[1:]]
            embeddings[title] = embedding

if FNAME_ENE:
    nline_fn = sum(1 for _ in tqdm(open(FNAME_ENE)))
    with open(FNAME_ENE) as fn:
        for line in tqdm(fn, total=nline_fn):
            enes, title = line.strip('\n').split('\t')
            title = title.replace(' ' , '_')
            enes = enes.split(';')
            eness[title] = enes


featuress = []
feature_counter = Counter()
for title in tqdm(titles):
    features = []

    features += ['TW1_{}'.format(unigram) for unigram in ngrams(title, n=1, unit='word')]
    features += ['TW2_{}'.format(bigram) for bigram in ngrams(title, n=2, unit='word')]
    features += ['TP2_{}'.format(pos_bigram) for pos_bigram in ngrams(title, n=2, unit='pos')]
    if nouns(title):
        features += ['TLN_{}'.format(nouns(title)[-1])]
    features += ['TCL1_{}'.format(title[-1])]
    features += ['TTL1_{}'.format(char_type(title[-1]))]
    if len(title) >= 2:
        features += ['TCL2_{}'.format(title[-2:])]
    if len(title) >= 3:
        features += ['TCL3_{}'.format(title[-3:])]
    
    first_sentence = first_sentences.get(title, '')
    if nouns(first_sentence):
        features += ['SLN_{}'.format(nouns(first_sentence)[-1])]
    else:
        features += ['SLN_']

    match_np = regex_np.search(first_sentence)
    if match_np:
        features += ['SLNP_{}'.format(match_np.group(1))]
    else:
        features += ['SLNP_']
    
    headings = headingss.get(title, [])
    if headings:
        features += ['H_{}'.format(heading) for heading in headings]
    else:
        features += ['H_']

    categories = categoriess.get(title, [])
    category_nouns = []
    for category in categories:
        if nouns(category):
            category_nouns += ['CLN_{}'.format(nouns(category)[-1])]
        # parent_categories = categoriess.get(category, [])
        # for parent_category in parent_categories:
        #     if nouns(parent_category):
        #         features += ['PCLN_{}'.format(nouns(parent_category)[-1])]
    if category_nouns:
        features += category_nouns
    else:
        features += ['CLN_']
    featuress.append(features)
    feature_counter.update(features)

try:
    id2feature = json.load(open(FNAME_ID2FEATURE))
except FileNotFoundError as e:
    id2feature = [feature for feature, _ in feature_counter.most_common(n=N_FEATURE)]
    json.dump(id2feature, open(FNAME_ID2FEATURE, 'w'), ensure_ascii=False)

feature2id = {feature: i for i, feature in enumerate(id2feature)}

assert len(titles) == len(featuress), (len(titles), len(featuress))
with open(FNAME_OUT, 'w') as fo:
    for title, features in tqdm(zip(titles, featuress)):
        j = {
            'title': title,
            'pageid': pageids[title],
            'features': features,
            'feature_ids': [feature2id[f] for f in features if f in feature2id]
        }
        if title in embeddings:
            j['embedding'] = embeddings[title]
        if title in eness:
            j['enes'] = eness[title]
        
        fo.write(json.dumps(j, ensure_ascii=False) + '\n')
