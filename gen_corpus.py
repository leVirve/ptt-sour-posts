import json
import itertools

from pymongo import MongoClient
import jieba


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
    corpus_output = f'data/corpus_{board}.txt'

    tokenizer = Tokenizer()
    db = PttDatabase(board)

    corpus = Corpus(corpus_output)

    for i, post in enumerate(db.posts()):
        corpus.write_words(tokenizer.segment(post['content']))
        for comment in post['comments']:
            corpus.write_words(tokenizer.segment(comment['content']))
        print(f'{board}, 處理第 {i} 篇文章，共 %d 篇' % len(db))

    corpus.close()
