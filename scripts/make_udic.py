import argparse


def main(args):
    with open(args.out, 'w') as fo:
        for i in range(args.tokens):
            token = f'<{i}>'
            line = f'{token},,,1,記号,一般,*,*,*,*,*,*,*'
            print(line, file=fo)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=1000)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    main(args)
