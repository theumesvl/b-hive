import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

path = "/eos/home-j/jusong/analysis/b-hive/output/InferenceTask/ttcc/Oct25v2_train/Oct25v2_test/Octv2_trainv2/ParticleTransformer2_ttcc/epochs_30/nominal/test_attack_nominal/"
truth_file = path+"truth.npy"
pred_file = path+"prediction.npy"


truth = np.load(truth_file)
plt.hist(truth, bins=[0,1,2,3,4,5])
plt.show()
pred = np.load(pred_file)
predi = np.argmax(pred, axis=1)

con = confusion_matrix(truth, predi,normalize="true" )
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = con, display_labels = ["ttbb", "ttbj", "ttcc", "ttcj", "ttother"])
cm_display.plot()
plt.show()
#cm_display.plot()
#plt.savefig('con.pdf', bbox_inches='tight')
con = confusion_matrix(truth, predi,normalize="pred" )
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = con, display_labels = ["ttbb", "ttbj", "ttcc", "ttcj", "ttother"])
cm_display.plot()
plt.show()
