import gensim
from special_string import *


def load_word2vec_model(path):
    # 对.sava保存的模型的加载：
    model = gensim.models.Word2Vec.load(path)
    # 对..wv.save_word2vec_format保存的模型的加载：
    # model = gensim.model.wv.load_word2vec_format('模型文件名')
    return model


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.font_manager as fm
    my_font = fm.FontProperties(fname=r'C:\Windows\Fonts\simkai.ttf')
    # model = load_word2vec_model(word2vec_model_path_zhwiki_update_814)
    # print(model.most_similar(''))
    pdf = PdfPages(ROOT_PATH + '\\latex\\svm.pdf')
    plt.figure(figsize=(8, 5))
    xfit = np.linspace(-1, 3.5)
    from sklearn.datasets.samples_generator import make_blobs  # 类数据的生成

    X, y = make_blobs(n_samples=50, n_features=2, centers=2,
                      random_state=0, cluster_std=0.60)
    # print(y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='PiYG_r')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit)
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='blue',
                         color='#AAAAAA', alpha=0.5)
    plt.legend(labels=['超平面1', '最佳超平面', '超平面2'], prop=my_font)
    plt.xlim(-1, 3.5)
    # plt.show()
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()

