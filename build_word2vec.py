import click
import jieba
from pymongo import MongoClient
from gensim import models


class Word2VecModel():

    name = 'output/%s-%d.model.bin'

    def __init__(self, filename):
        self.filename = filename

    def train(self, corpus_file, encode_size=250):
        sentences = models.word2vec.Text8Corpus(corpus_file)
        model = models.word2vec.Word2Vec(sentences, size=encode_size, workers=8)
        model.save(self.name % (self.filename, encode_size))

    def load(self, encode_size=250):
        return models.Word2Vec.load(self.name % (self.filename, encode_size))


class Corpus():

    def __init__(self, filename):
        self.writer = open(filename, 'w', encoding='utf8')

    def write_words(self, words):
        self.writer.write(' '.join(words) + ' ')

    def close(self):
        self.writer.close()


class Tokenizer():

    dictionary_file = 'data/dict.txt.big'
    stopwords_file = 'data/stop_words.txt'

    def __init__(self):
        jieba.set_dictionary(self.dictionary_file)
        self.stopwords = self.get_stopwords()

    def get_stopwords(self):
        with open(self.stopwords_file, 'r', encoding='utf-8') as f:
            return set([line.strip('\n') for line in f])

    def segment(self, text):
        words = jieba.cut(text, cut_all=False)
        return [w for w in words if w not in self.stopwords and w.strip()]


class PttDatabase():

    def __init__(self, board):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client['ptt']
        self.collection = self.db[board]

    def posts(self):
        for post in self.collection.find():
            yield post

    def __len__(self):
        return self.collection.count()


def create_corpus(board, corpus_file):
    db = PttDatabase(board)
    corpus = Corpus(corpus_file)
    tokenizer = Tokenizer()
    for i, post in enumerate(db.posts(), start=1):
        corpus.write_words(tokenizer.segment(post['content']))
        for comment in post['comments']:
            corpus.write_words(tokenizer.segment(comment['content']))
        print('%s, 處理第 %d 篇文章，共 %d 篇' % (board, i, len(db)))
    corpus.close()


@click.command()
@click.option('--board', default='baseball', help='選擇資料來源看板')
@click.option('--train', is_flag=True, help='是否訓練 word2vec')
@click.option('--encode_size', default=250, help='word2vec 向量長度')
@click.option('--text', default='', help='測試字串')
def main(board, train, encode_size, text):
    corpus_file = 'output/corpus_%d.txt' % board
    w2v = Word2VecModel(board)

    if train:
        create_corpus(board, corpus_file)
        w2v.train(corpus_file, encode_size=encode_size)

    model = w2v.load(encode_size=encode_size)
    res = model.most_similar(text.strip(), topn=20)
    vec = model[text]
    print(vec, '\n', res)


if __name__ == '__main__':
    main()
