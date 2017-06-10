import click
from build_word2vec import Word2VecModel


@click.command()
@click.option('--board', default='baseball', help='選擇資料來源看板')
def main(board):
    w2v = Word2VecModel(board)

    model = w2v.load(encode_size=250)

    while True:
        try:
            text = input('Word > ')
            if not text:
                break
            result = model.most_similar(text.strip(), topn=20)
            for r in result:
                print(r)
        except:
            print('Cannot found', text)

if __name__ == '__main__':
    main()
