import jieba
from pymongo import MongoClient
from gensim.models import word2vec


class Word2VecModel():

    name = 'output/%s-%d.model.bin'

    def __init__(self, filename, corpus_file, encode_size=250):
        self.filename = filename
        self.corpus_file = corpus_file
        self.encode_size = encode_size

    def train(self):
        sentences = word2vec.Text8Corpus(self.corpus_file)
        model = word2vec.Word2Vec(sentences, size=self.encode_size, workers=8)
        model.save(self.name % (self.filename, self.encode_size))


class Corpus():

    def __init__(self, filename):
        self.writer = open(corpus_output, 'w', encoding='utf8')

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


if __name__ == '__main__':

    board = 'baseball'
    corpus_file = f'output/corpus_{board}.txt'

    db = PttDatabase(board)

    corpus = Corpus(corpus_file)
    tokenizer = Tokenizer()
    for i, post in enumerate(db.posts()):
        corpus.write_words(tokenizer.segment(post['content']))
        for comment in post['comments']:
            corpus.write_words(tokenizer.segment(comment['content']))
        print(f'{board}, 處理第 {i} 篇文章，共 %d 篇' % len(db))
    corpus.close()

    model = Word2VecModel(board, corpus_file, encode_size=250)
    model.train()
