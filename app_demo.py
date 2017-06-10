import pickle

from flask import Flask
from flask import request, render_template

from build_word2vec import Word2VecModel, Tokenizer
from crawler import core


app = Flask(__name__)
tokenizer = Tokenizer()
spider = core.PTTSpider()
result_map = {0: '中性', 1: '酸文'}


def load_wor2vec_model(board):
    w2v = Word2VecModel(board)
    return w2v.load(encode_size=250)


def load_tfidf_model(method, board):
    with open('output/%s-%s-tfidf.pickle' % (board, method), 'rb') as f:
        return pickle.load(f)


w2v_model_zoo = {
    'LoL': load_wor2vec_model('lol'),
    'Movie': load_wor2vec_model('movie'),
    'Baseball': load_wor2vec_model('baseball'),
}

tfidf_model_zoo = {
    'LoL': load_tfidf_model('nb', 'lol'),
    'Movie': load_tfidf_model('nb', 'movie'),
    'Baseball': load_tfidf_model('nb', 'baseball'),
}

svm_model_zoo = {
    'LoL': load_tfidf_model('svm', 'lol'),
    'Movie': load_tfidf_model('svm', 'movie'),
    'Baseball': load_tfidf_model('svm', 'baseball'),
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/word2vec', methods=['GET', 'POST'])
def word2vec():
    if request.method == 'POST':
        try:
            board_name = request.form['board_name']
            keyword = request.form['keyword']

            model = w2v_model_zoo[board_name]
            results = model.most_similar(keyword.strip(), topn=20)
        except:
            results = None

        return render_template(
            'word2vec.html',
            show=True,
            board_name=board_name, keyword=keyword, results=results)
    else:
        return render_template('word2vec.html')


@app.route('/predict_baseline', methods=['GET', 'POST'])
def nb_predict():

    if request.method == 'POST':
        board_name = request.form['board_name']
        method_name = request.form['method_name']
        keyword = request.form['keyword']

        if method_name == 'SVM':
            model = svm_model_zoo[board_name]
        else:
            model = tfidf_model_zoo[board_name]

        if keyword.startswith('http'):
            post = spider.get_post(keyword)
            comments = [cmt['comment'] for cmt in post['comments']]
            keyword = post['content'] + ' '.join(comments)

        input_corpus = tokenizer.segment(keyword)
        result = model.predict([' '.join(input_corpus)])[0]
        # print(input_corpus)
        return render_template(
            'predict_baseline.html',
            show=True,
            board_name=board_name, keyword=keyword, result=result_map[result])
    else:
        return render_template('predict_baseline.html')


if __name__ == "__main__":
    app.run(debug=True)
