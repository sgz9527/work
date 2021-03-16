# coding=utf-8

"""
@File  : text_clustering.py
@Author: Xu Qiqiang
@Date  : 2020/11/11 0011
"""
from load_resume_data import load_data_for_single_muti_classification
from feature_engineer import trans_to_wordvec_by_word2vec
from special_string import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data_for_single_muti_classification(data_set=1, train_num=100, test_num=10)
    train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                                           word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='full')
    # pca = PCA(n_components=10)
    # train_x = pca.fit_transform(train_x)
    model = KMeans(n_clusters=10)
    model.fit(train_x)
    res = model.predict(train_x)
    # print(train_y)
    # print(adjusted_mutual_info_score(train_y, res))
    print(silhouette_score(train_x, res))



