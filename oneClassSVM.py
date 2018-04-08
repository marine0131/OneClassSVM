import numpy as np
import json
import cPickle as pickle
from sklearn import svm

# xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# # Generate train data
# X = 0.3 * np.random.randn(100, 2)
# X_train = np.r_[X + 2, X - 2]
# # Generate some regular novel observations
# X = 0.3 * np.random.randn(20, 2)
# X_test = np.r_[X + 2, X - 2]
# # Generate some abnormal novel observations
# X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#
# # fit the model
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(X_outliers)
# n_error_train = y_pred_train[y_pred_train == -1].size
# n_error_test = y_pred_test[y_pred_test == -1].size
# n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
#
# # plot the line, the points, and the nearest vectors to the plane
# Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
#
# plt.title("Novelty Detection")
# plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
# a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')
#
# s = 40
# b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
# b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
#                  edgecolors='k')
# c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
#                 edgecolors='k')
# plt.axis('tight')
# plt.xlim((-5, 5))
# plt.ylim((-5, 5))
# plt.legend([a.collections[0], b1, b2, c],
#            ["learned frontier", "training observations",
#             "new regular observations", "new abnormal observations"],
#            loc="upper left",
#            prop=matplotlib.font_manager.FontProperties(size=11))
# plt.xlabel(
#     "error train: %d/200 ; errors novel regular: %d/40 ; "
#     "errors novel abnormal: %d/40"
#     % (n_error_train, n_error_test, n_error_outliers))
# plt.show()

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    x = []
    for l in lines:
        x.append(json.loads(l)['feature'])

    x = np.array(x)
    y = np.c_[x[:,1], x[:,3:6], x[:,12:14], x[:,16:18], x[:,24:26], x[:, 27:30]]
    # y = x
    return y

# train_set is  samples(n)xfeatures(m) numpy array
def train(train_set):
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
    clf.fit(train_set)

    return clf


if __name__ == "__main__":
    # read dataset
    X1 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0328_part1.log")
    X2 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0328_part2.log")
    X3 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0328_part3.log")
    X4 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0329.log")
    X5 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0404.log")
    C1 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/pos_0319.log")
    X = np.r_[X1, X2, X3, X4, X5]

    # train
    print('start trainning')
    clf = train(X)
    print('done trainning')
    # with open("./model/new_model", "wb") as f:
    #     pickle.dump(itree, f)

    # with open("./model", "r") as f:
    #     itree =  pickle.load(f)

    x_predict = clf.predict(X)
    print('train error: {}'.format(x_predict[x_predict == -1].size/float(x_predict.size)))

    # test
    t1 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0313.log")
    t2 = load_data("/home/whj/gitrepo/audioAnalysis/feature_data/neg_0316.log")
    t = np.r_[t1, t2]
    t_predict = clf.predict(t)
    print('test error: {}'.format(t_predict[t_predict == -1].size/float(t_predict.size)))

    ab1= load_data("/home/whj/gitrepo/audioAnalysis/feature_data/pos_0313.log")
    ab2= load_data("/home/whj/gitrepo/audioAnalysis/feature_data/pos_0319.log")
    ab = np.r_[ab1, ab2]
    ab_predict = clf.predict(ab)
    print('abnormal error: {}'.format(ab_predict[t_predict == 1].size/float(ab_predict.size)))

    with open("./result.txt", 'wb') as f:
        tt1 = t_predict
        tt2 = ab_predict
        # np.savetxt(f, tt)
        # f.write('\n\n')
        np.savetxt(f, tt1)
        f.write('\n\n')
        np.savetxt(f, tt2)
