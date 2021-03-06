# -------------简历数据根目录----------------------------------- #
ROOT_PATH = "D:\\resume_proj"


# -------------原始简历文本目录----------------------------------- #
TRAIN_TXT_PATH = ROOT_PATH + '\\material\\TrainTxtResume'
TEST_TXT_PATH = ROOT_PATH + '\\material\\TestTxtResume'
MATERIAL_TRAIN_PATH = ROOT_PATH + '\\material\\TrainResume'
MATERIAL_TEST_PATH = ROOT_PATH + '\\material\\TestResume'
DOWNLOAD_RESUME_PATH = ROOT_PATH + '\\material\\download_resume'
DOWNLOAD_RESUME_PATH1 = ROOT_PATH + '\\material\\download_resume1'


# -------------多分类模型训练数据集所在路径----------------------------------- #
MUTI_CLASSFICATION_DATA_PATH = ROOT_PATH + '\\muti_classification_data'
MUTI_CLASSFICATION_TRAIN_DATA_PATH = MUTI_CLASSFICATION_DATA_PATH + '\\train_data'
MUTI_CLASSFICATION_TEST_DATA_PATH = MUTI_CLASSFICATION_DATA_PATH + '\\test_data'
TEST_PATH = ROOT_PATH+"\\muti_classification_data\\test_data"
TRAIN_PATH = ROOT_PATH+"\\muti_classification_data\\train_data"
RNN_DATA_PATH = ROOT_PATH + '\\rnn_data'
RNN_DATA_DATA_PATH1 = RNN_DATA_PATH + '\\data1'
RNN_DATA_LABEL_PATH1 = RNN_DATA_PATH + '\\label1'
RNN_DATA_DATA_PATH2 = RNN_DATA_PATH + '\\data2'
RNN_DATA_LABEL_PATH2 = RNN_DATA_PATH + '\\label2'
RNN_DATA_DATA_PATH3 = RNN_DATA_PATH + '\\data3'
RNN_DATA_LABEL_PATH3 = RNN_DATA_PATH + '\\label3'
RNN_ABNORMAL_DATA_DATA_PATH = RNN_DATA_PATH + '\\abnormal_data'
RNN_ABNORMAL_DATA_LABEL_PATH = RNN_DATA_PATH + '\\abnormal_label'
RNN_BLOCK_DATA_PATH = RNN_DATA_PATH + '\\block_data'
#----------------标题识别模型训练数据所在路径-----------------------------#
TITLE_REC_DATA_PATH = ROOT_PATH + '\\title_rec_data'

# -------------辅助资料所在路径----------------------------------- #
USER_DIC_PATH = ROOT_PATH + '\\material\\user_dic.txt'
STOP_LIST_PATH = ROOT_PATH + '\\material\\stop_list.txt'
OTHER_TXT_PATH = ROOT_PATH + '\\material\\other.txt'
COMPANY_TXT_PATH = ROOT_PATH + '\\company.txt'
TITLE_LIST_PATH = ROOT_PATH + '\\material\\title_list.txt'
WEBDRIVER_PATH = r'D:\Anaconda3\envs\tf\Scripts\chromedriver.exe'
ITEM_EXP_TXT_PATH = ROOT_PATH + '\\material\\item_exp.txt'
TITLE_ORDER_PATH = ROOT_PATH + '\\title_order.txt'


# -------------word2vec语料所在路径----------------------------------- #
it_corpus_path = './blog'
new_it_corput_path = './blog_new'
it_corpus_name = 'it_corpus.txt'
zhwiki_corpus_path = ROOT_PATH + '\\zhwiki-latest-pages-articles.xml.bz2'
sentence_zhwiki_corpus_txt_path = ROOT_PATH + '\\zhwiki-latest-pages-articles_txt.txt'
sentence_it_corpus_path = new_it_corput_path + '/' + it_corpus_name
sentence_51job_resume_corpus_path = ROOT_PATH + '\\sentence_51job_resume_corpus.txt'
sentence_ch_wiki_path = ROOT_PATH + '\\' + 'word2vec_corpus_chinese_wiki.txt'
sentence_train_data_path = ROOT_PATH + '\\' + 'word2vec_corpus_train_data.txt'
sentence_rnn_data_word2vec_corpus_path = ROOT_PATH + '\\' + 'rnn_data_word2vec_corpus_txt.txt'
word2vec_model_path_train = ROOT_PATH + '\\word2vec'
word2vec_model_path_wiki = ROOT_PATH + '\\word2vec_add_wiki'
word2vec_model_path_it = ROOT_PATH + '\\word2ved_it'
word2vec_model_path_it_new = ROOT_PATH + '\\word2vec_it_new'
word2vec_model_path_it_train = ROOT_PATH + '\\word2ved_it_train'
word2vec_model_path_update_8_13 = ROOT_PATH + '\\word2vec_update_8_13'
word2vec_model_path_zhwiki_update_814 = ROOT_PATH + '\\word2vec_update_814_chwiki'
word2vec_model_path_zhwiki_rnn_update_20_923 = ROOT_PATH + '\\word2vec_update_20_923_chwiki_rnn'
word2vec_model_path_2021_2_5 = ROOT_PATH + '\\word2vec_update_2021_2_5'
# ---------------------多分类模型所在路径--------------------------- #
muti_full_tfidf_model_path = ROOT_PATH + '\\material\\ts_muti_nn_model5.h5'
muti_full_word2vec_model_path = ROOT_PATH + '\\material\\ts_muti_nn_model_word2vec.h5'
muti_textcnn_api_model_path = ROOT_PATH + '\\material\\text_cnn_model_word2vec_API_mode.h5'
muti_textcnn_api_model_update_path = ROOT_PATH + '\\material\\text_cnn_model_word2vec_API_mode_update.h5'
muti_textcnn_api_model_update1_path = ROOT_PATH + '\\material\\text_cnn_model_word2vec_API_mode_update1.h5'
muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec = ROOT_PATH + '\\material\\text_cnn_model_word2vec_API_mode_update2_zhwiki_word2vec.h5'
FNN_MODEL_PATH = ROOT_PATH + '\\fnn_11_30'
BBRNN_MODEL_PATH = ROOT_PATH + '\\b_brnn_model_11-29'
HYBRID_MODEL_DYNAMIC_WEIGHT_PATH = ROOT_PATH + '\\hybrid_model_dynamic_weight_12_1'
BBRNN_IMPROVED_BY_FEATURE_INTEGRATION_PATH = ROOT_PATH + '\\B-BRNN-IMPROVED-BY-FEATURE-INTEGRATION'

FNN_100_PATH = ROOT_PATH + '\\fnn_100'
FNN_200_PATH = ROOT_PATH + '\\fnn_200'
FNN_300_PATH = ROOT_PATH + '\\fnn_300'
FNN_400_PATH = ROOT_PATH + '\\fnn_400'
FNN_500_PATH = ROOT_PATH + '\\fnn_500'
FNN_600_PATH = ROOT_PATH + '\\fnn_600'
FNN_700_PATH = ROOT_PATH + '\\fnn_700'
JOINT_100_PATH = ROOT_PATH + '\\joint_100'
JOINT_200_PATH = ROOT_PATH + '\\joint_200'
JOINT_300_PATH = ROOT_PATH + '\\joint_300'
JOINT_400_PATH = ROOT_PATH + '\\joint_400'
JOINT_500_PATH = ROOT_PATH + '\\joint_500'
JOINT_600_PATH = ROOT_PATH + '\\joint_600'
JOINT_700_PATH = ROOT_PATH + '\\joint_700'
BRNN_100_PATH = ROOT_PATH + '\\brnn_100'
BRNN_200_PATH = ROOT_PATH + '\\brnn_200'
BRNN_300_PATH = ROOT_PATH + '\\brnn_300'
BRNN_400_PATH = ROOT_PATH + '\\brnn_400'
BRNN_500_PATH = ROOT_PATH + '\\brnn_500'
BRNN_600_PATH = ROOT_PATH + '\\brnn_600'
BRNN_700_PATH = ROOT_PATH + '\\B-BRNN-IMPROVED-BY-FEATURE-INTEGRATION'
HYBRID_100_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model_100'
HYBRID_200_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model_200'
HYBRID_300_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model_300'
HYBRID_400_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model_400'
HYBRID_500_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model_500'
HYBRID_600_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model_600'
HYBRID_700_PATH = ROOT_PATH + '\\dynamic_decision_hybrid_model'
DW_HYBRID_100_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_100'
DW_HYBRID_200_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_200'
DW_HYBRID_300_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_300'
DW_HYBRID_400_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_400'
DW_HYBRID_500_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_500'
DW_HYBRID_600_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_600'
DW_HYBRID_700_PATH = ROOT_PATH + '\\dynamic_weight_hybrid_model_700'
MUTUAL_LEARNING_MODEL_700_PATH = ROOT_PATH + '\\teacher_student_model_700'
KNOWLEDGE_DISTILL_MODEL_700_PATH = ROOT_PATH + '\\knowledge_distill_model_700'
NOISE_10_BRNN_PATH = ROOT_PATH + '\\noise_10_brnn_model'
NOISE_9_BRNN_PATH = ROOT_PATH + '\\noise_9_brnn_model'
NOISE_8_BRNN_PATH = ROOT_PATH + '\\noise_8_brnn_model'
NOISE_7_BRNN_PATH = ROOT_PATH + '\\noise_7_brnn_model'
NOISE_6_BRNN_PATH = ROOT_PATH + '\\noise_6_brnn_model'
NOISE_5_BRNN_PATH = ROOT_PATH + '\\noise_5_brnn_model'
NOISE_4_BRNN_PATH = ROOT_PATH + '\\noise_4_brnn_model'
NOISE_3_BRNN_PATH = ROOT_PATH + '\\noise_3_brnn_model'
NOISE_2_BRNN_PATH = ROOT_PATH + '\\noise_2_brnn_model'
NOISE_1_BRNN_PATH = ROOT_PATH + '\\noise_1_brnn_model'
# --------------------------------------序列标注模型路径-------------------------------------------------#
JOB_NAME_EXTRACTION_MODEL_PATH = ROOT_PATH + '\\job_name_extraction_model'

# ---------------------------------------------------------匹配模式--------------------------------------------------#
DEL_SPECIAL_PAT = r'[^\u4E00-\u9FD5\d\w\]\[://@.]+'  # 除去中文、数字、英文字母、一些有用标识符以外的字符
DEL_SPECIAL_PAT2 = r'[^\u4E00-\u9FD5\d\w\]\[:：（）()_-——//@.]+'
CORRECT_NAME = r'[a-zA-Z]+'  # 检测是否存在多个英文字符
DEL_SPECIAL_PAT1 = r'[^\u4E00-\u9FD5\d\w\]\[://@. \n]+'
CORRECT_LINK = r'([a-zA-Z]+://[^\s]*[.com|.cn|.edu|.org|.vip|.top|.net](/[a-zA-Z\d_]+)*)'
CORRECT_LINK1 = r'([wW]{3}.[^\s]*[.com|.cn|.edu|.org|.vip|.top|.net](/[a-zA-Z\d_]+)*)'
