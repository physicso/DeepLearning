import pandas
import matplotlib.pyplot as plt
from sklearn import metrics
import config

aucs = []


def auc(actual, pred):
    fpr, tpr, _ = metrics.roc_curve(actual, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


def roc_plot(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


for index in range(config.SAVE_INTERVAL, config.TRAIN_ITERATION + 1, config.SAVE_INTERVAL):
    data = pandas.read_csv('resnet-34/%s.csv' % str(index), header=None)
    label, probability = data[1], data[2]
    auc_value = auc(label, probability)
    aucs.append(auc_value)

plt.plot(aucs)
plt.show()

fpr, tpr, thresholds = metrics.roc_curve(label, probability, pos_label=1)
print(fpr, tpr, thresholds)
roc_plot(fpr, tpr)
