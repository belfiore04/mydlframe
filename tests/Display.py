import base64
import json
from collections import Counter
from io import BytesIO

from hmmlearn import hmm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dlframe import WebManager, Logger
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from hmmlearn.hmm import GaussianHMM
import nltk
from nltk.corpus import brown
import math
import numpy as np

#
# # 数据集
# class IrisDataset:
#     def __init__(self) -> None:
#         super().__init__()
#         iris = load_iris()
#         self.data = iris.data
#         self.target = iris.target
#         self.logger = Logger.get_logger('IrisDataset')
#         self.logger.print("I'm iris dataset")
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __getitem__(self, idx: int):
#         return self.data[idx], self.target[idx]
#
#
# class WineDataset:
#     def __init__(self) -> None:
#         super().__init__()
#         wine = load_wine()
#         self.data = wine.data
#         self.target = wine.target
#         self.logger = Logger.get_logger('WineDataset')
#         self.logger.print("I'm wine dataset")
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __getitem__(self, idx: int):
#         return self.data[idx], self.target[idx]
#
#
# # class BrownDataset:
# #     def __init__(self, category='news', num_sentences=100, min_freq=1, oov_token='<UNK>'):
# #         self.category = category
# #         self.num_sentences = num_sentences
# #         self.min_freq = min_freq
# #         fileids=brown.fileids()
# #         #self.target = [brown.categories(fileid)[0]for fileid in fileids]
# #         self.oov_token = oov_token
# #         self.logger = Logger.get_logger('DataLoader')
# #         self.data = []
# #         self.labels = []
# #         self.word_to_int = {}
# #         self.int_to_word = {}
# #         self.processed_data = []
# #         self.lengths = []
# #         self.load_data()
# #         self.build_vocab()
# #         self.encode_data()
# #
# #     def load_data(self):
# #         sentences = brown.sents(categories=self.category)[:self.num_sentences]
# #         self.data = sentences
# #         self.labels = [self.get_label(sentence) for sentence in sentences]
# #         self.logger.info(f"Loaded {len(self.data)} sentences from category '{self.category}'.")
# #
# #     def get_label(self, sentence):
# #         # 示例：为每个句子分配一个随机的隐藏状态标签
# #         # 实际应用中，这可能基于某些规则或外部标签
# #         return [np.random.randint(0, 3) for _ in sentence]  # 假设有3个隐藏状态
# #
# #     def build_vocab(self):
# #         counter = Counter(word.lower() for sentence in self.data for word in sentence)
# #         vocab = {word for word, freq in counter.items() if freq >= self.min_freq}
# #         self.word_to_int = {word: i for i, word in enumerate(vocab, start=1)}  # 0 reserved for OOV
# #         self.word_to_int[self.oov_token] = 0
# #         self.int_to_word = {i: word for word, i in self.word_to_int.items()}
# #         self.logger.info(f"Built vocabulary of size {len(self.word_to_int)} (including OOV).")
# #
# #     def encode_data(self):
# #         self.processed_data = [
# #             [self.word_to_int.get(word.lower(), self.word_to_int[self.oov_token]) for word in sentence]
# #             for sentence in self.data
# #         ]
# #         self.lengths = [len(sentence) for sentence in self.processed_data]
# #         self.logger.info("Encoded data into integer sequences with OOV handling.")
# #
# #     def get_sequences(self):
# #         # 返回合并后的观测序列和长度
# #         concatenated = np.concatenate(self.processed_data).reshape(-1, 1)
# #         return concatenated, self.lengths
#
# def plot_metrics(accuracy, precision, recall, ari):
#     labels = ['Accuracy', 'Precision', 'Recall', 'ARI']
#     values = [accuracy, precision, recall, ari]
#
#     plt.figure(figsize=(8, 6))
#     plt.bar(labels, [v for v in [accuracy, precision, recall, ari] if v is not None], color='blue')
#     plt.ylabel('Scores')
#     plt.title('Model Performance Metrics')
#     plt.ylim(0, 1.05)  # 假设所有值都在0到1之间
#
#     # 将图像保存到BytesIO对象中
#     img_buffer = BytesIO()
#     plt.savefig(img_buffer, format='png')
#     img_buffer.seek(0)
#     img_data = img_buffer.getvalue()
#
#     # 将图像数据转换为Base64
#     img_base64 = base64.b64encode(img_data).decode('utf-8')
#     plt.close()
#
#     return img_base64
#
#
# # 数据集切分器
# class TestSplitter:
#     def __init__(self, ratio) -> None:
#         super().__init__()
#         self.ratio = ratio
#         self.logger = Logger.get_logger('TestSplitter')
#         self.logger.print("I'm ratio:{}".format(self.ratio))
#
#     def split(self, dataset):
#         data, target = dataset.data, dataset.target
#         indices = np.arange(len(data))
#         np.random.shuffle(indices)
#
#         split_point = math.floor(len(data) * self.ratio)
#         train_indices = indices[:split_point]
#         test_indices = indices[split_point:]
#
#         trainingSet = [(data[i], target[i]) for i in train_indices]
#         testingSet = [(data[i], target[i]) for i in test_indices]
#
#         self.logger.print("split!")
#         self.logger.print("training_len = {}".format(len(trainingSet)))
#         self.logger.print("testing_len = {}".format(len(testingSet)))
#         return trainingSet, testingSet
#
#
# # 模型
# class KNN:
#     def __init__(self, k_neighbors=3) -> None:
#         super().__init__()
#         self.k_neighbors = k_neighbors
#         self.logger = Logger.get_logger('KNN')
#         self.model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, k_neighbors = {}, trainDataset = {}".format(self.k_neighbors, trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class NaiveBayes:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('NaiveBayes')
#         self.model = GaussianNB()
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, trainDataset = {}".format(trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class DecisionTree:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('DecisionTree')
#         self.model = DecisionTreeClassifier()
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, trainDataset = {}".format(trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class SVM:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('SVM')
#         self.model = SVC()
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, trainDataset = {}".format(trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class LogisticRegressionModel:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('LogisticRegression')
#         self.model = LogisticRegression()
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, trainDataset = {}".format(trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class MaxEntropy:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('MaxEntropy')
#         self.model = LogisticRegression()
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, trainDataset = {}".format(trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class AdaBoost:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('AdaBoost')
#         self.model = AdaBoostClassifier()
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         y_train = np.array([data[1] for data in trainDataset])
#         self.model.fit(X_train, y_train)
#         self.logger.print("training, trainDataset = {}".format(trainDataset))
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# class EMAlgorithm:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('EMAlgorithm')
#         self.model = GaussianMixture(n_components=3)  # 指定聚类数
#
#     def train(self, trainDataset) -> None:
#         X_train = np.array([data[0] for data in trainDataset])
#         self.model.fit(X_train)
#         self.logger.print("Training completed.")
#
#     def test(self, testDataset):
#         X_test = np.array([data[0] for data in testDataset])
#         y_pred = self.model.predict(X_test)
#         self.logger.print("Testing completed.")
#         return y_pred  # 返回预测的聚类标签
#
#
# class HiddenMarkovModel:
#     def __init__(self, n_components=3, covariance_type='diag', n_iter=100, random_state=42):
#         self.logger = Logger.get_logger('HiddenMarkovModel')
#         self.model = hmm.GaussianHMM(
#             n_components=n_components,
#             covariance_type=covariance_type,
#             n_iter=n_iter,
#             random_state=random_state
#         )
#         self.logger.print(f"Initialized HMM with {n_components} components.")
#
#     def train(self, X, lengths):
#         self.model.fit(X, lengths)
#         self.logger.print("Training HMM completed.")
#
#     def predict(self, X, lengths):
#         y_pred = self.model.predict(X, lengths)
#         self.logger.print("Prediction with HMM completed.")
#         return y_pred
#
#     def score(self, X, lengths):
#         log_likelihood = self.model.score(X, lengths)
#         self.logger.print(f"Model log likelihood: {log_likelihood:.2f}")
#         return log_likelihood
#
#     def select_best_model(self, X, lengths, max_components=10):
#         bic_scores = []
#         aic_scores = []
#         for n in range(1, max_components + 1):
#             model = hmm.GaussianHMM(
#                 n_components=n,
#                 covariance_type='diag',
#                 n_iter=100,
#                 random_state=42
#             )
#             model.fit(X, lengths)
#             bic = model.bic(X)
#             aic = model.aic(X)
#             bic_scores.append(bic)
#             aic_scores.append(aic)
#             self.logger.print(f"n_components={n}, BIC={bic:.2f}, AIC={aic:.2f}")
#
#         # 选择 BIC/AIC 最低的模型
#         best_n_bic = np.argmin(bic_scores) + 1
#         best_n_aic = np.argmin(aic_scores) + 1
#         self.logger.print(f"Best number of components by BIC: {best_n_bic}")
#         self.logger.print(f"Best number of components by AIC: {best_n_aic}")
#         return best_n_bic, best_n_aic
#
# class KMeansModel:
#     def __init__(self, learning_rate) -> None:
#         super().__init__()
#         self.learning_rate = learning_rate
#         self.logger = Logger.get_logger('KMeansModel')
#         self.model = KMeans()
#
#     def train(self, trainDataset) -> None:
#         X_train = [data[0] for data in trainDataset]
#         self.model.fit(X_train,train_lenth=len(X_train))
#         self.logger.print("training, lr = {}, trainDataset = {}".format(self.learning_rate, trainDataset))
#
#     def test(self, testDataset):
#         X_test = [data[0] for data in testDataset]
#         y_hat = self.model.predict(X_test)
#         self.logger.print("testing")
#         return y_hat
#
#
# # 结果判别器
# class AccuracyJudger:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('TestJudger')
#
#     def judge(self, y_hat, test_dataset) -> None:
#         y_true = [data[1] for data in test_dataset]
#         accuracy = np.mean(np.array(y_hat) == np.array(y_true))
#         self.logger.print("y_hat = {}".format(y_hat))
#         self.logger.print("gt = {}".format(y_true))
#         self.logger.print("Accuracy = {:.2f}%".format(accuracy * 100))
#
#
# class PrecisionCalculator:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('PrecisionCalculator')
#
#     def judge(self, y_hat, test_dataset) -> None:
#         y_true = [data[1] for data in test_dataset]
#         precision = precision_score(y_true, y_hat, average='macro')
#         self.logger.print("y_hat = {}".format(y_hat))
#         self.logger.print("gt = {}".format(y_true))
#         self.logger.print("Precision = {:.2f}".format(precision))
#
#
# class RecallCalculator:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('RecallCalculator')
#
#     def judge(self, y_hat, test_dataset) -> None:
#         y_true = [data[1] for data in test_dataset]
#         recall = recall_score(y_true, y_hat, average='macro')
#         self.logger.print("y_hat = {}".format(y_hat))
#         self.logger.print("gt = {}".format(y_true))
#         self.logger.print("Recall = {:.2f}".format(recall))
#
#
# class AdjustedRandIndexCalculator:
#     def __init__(self) -> None:
#         super().__init__()
#         self.logger = Logger.get_logger('AdjustedRandIndexCalculator')
#
#     def judge(self, y_pred, test_dataset) -> None:
#         # 提取真实标签
#         y_true = np.array([data[1] for data in test_dataset])
#         # 计算 Adjusted Rand Index
#         ari_score = adjusted_rand_score(y_true, y_pred)
#         # 输出结果
#         self.logger.print("Adjusted Rand Index (ARI): {:.2f}".format(ari_score))

def training_position(pos):
    logger = Logger.get_logger('TrainingPosition')
    if pos == '本地':
        logger.print("training in local")
    elif pos == '远程':
        logger.print("training in remote")

if __name__ == '__main__':
    with WebManager(parallel=False) as manager:
        dataset = manager.register_element('数据集', {'鸢尾花':None, '红酒': None})
        splitter = manager.register_element('数据分割',
                                            {'ratio:0.8': None, 'ratio:0.5': None})
        model = manager.register_element('模型', {'KNN-3': None,'KNN-5':None, 'NaiveBayes': None,
                                                  'DecisionTree': None, 'SVM': None,
                                                  'LogisticRegression': None,
                                                  'MaxEntropy': None, 'AdaBoost': None,
                                                  'EMAlgorithm（聚类）': None,
                                                  'KMeansModel': None})
        judger = manager.register_element('评价指标', {'准确率': None, '精确率': None,
                                                       '召回率': None,
                                                       "ARI（聚类）": None})
        position = manager.register_element('训练位置', {'本地': training_position("本地"), '远程-控制': training_position("远程-控制"),'远程-计算': training_position("远程-计算")})
        train_data_test_data = splitter.split(dataset)
        train_data, test_data = train_data_test_data[0], train_data_test_data[1]
        model.train(train_data)
        y_hat = model.test(test_data)
        judger.judge(y_hat, test_data)
