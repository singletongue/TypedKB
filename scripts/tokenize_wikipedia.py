import re
import json
import argparse
from urllib.parse import unquote
from collections import OrderedDict

import MeCab
from tqdm import tqdm


regex_digits = re.compile(r'\d+')
regex_hyperlink = re.compile(r'<a href="(.+?)">(.+?)</a>')


def main(args):
    mt = MeCab.Tagger(f'-O wakati -u {args.udic}')

    print('counting # of articles')
    n_articles = sum(1 for _ in open(args.extract))
    print(f'{n_articles} articles')

    print('processing articles')
    with open(args.extract) as fi, open(args.out, 'w') as fo:
        for line in tqdm(fi, total=n_articles, ncols=60):
            article = json.loads(line)
            title = article['title']
            text = article['text']
            text = text[text.find('\n\n')+2:]

            hyperlinks = dict()
            hyperlinks[title] = title
            for match in regex_hyperlink.finditer(text):
                dst = unquote(match.group(1))
                anchor = match.group(2)
                if regex_digits.match(anchor):
                    continue

                if '#' in dst:
                    dst = dst[:dst.find('#')]

                if len(dst) > 0:
                    hyperlinks.setdefault(anchor, dst)

            text = regex_hyperlink.sub(r'\2', text)

            hyperlinks_sorted = OrderedDict(sorted(
                hyperlinks.items(), key=lambda t: len(t[0]), reverse=True))

            for idx, hyperlink in enumerate(hyperlinks_sorted.items()):
                anchor, dst = hyperlink
                idx_token = f' <{idx}> '
                text = text.replace(anchor, idx_token)

            text_tokenized = mt.parse(text).strip()

            for idx, hyperlink in enumerate(hyperlinks_sorted.items()):
                anchor, dst = hyperlink
                idx_token = f'<{idx}>'
                hyperlink_token = f'[{dst}]'
                text_tokenized = text_tokenized.replace(
                                                idx_token, hyperlink_token)

            print(text_tokenized, file=fo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', type=str, required=True)
    parser.add_argument('--udic', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    main(args)
