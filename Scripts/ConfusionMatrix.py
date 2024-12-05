import Main
import sklearn.metrics as metrics
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


pred_binary = np.where(Main.model.predict(Main.X_test) > 0.5, 1, 0).flatten() # convert to binary prediction; treshold=0.5

print ('\n>>  The size of the independent test dataset is: ', len(Main.Y_test))
print('\>>  Numbers of labels in targets are:')
print('0: ', Counter(Main.Y_test)[0])
print('1: ', Counter(Main.Y_test)[1])

print('\n>>  Numbers of labels in the binary predicted output are:')
print('0: ', Counter(pred_binary)[0])
print('1: ', Counter(pred_binary)[1])

print('\n>>  Metrics classification report:\n')
print(metrics.classification_report(Main.Y_test, pred_binary))

confusion_matrix_simple = metrics.confusion_matrix(y_true = Main.Y_test, y_pred = pred_binary) # build confusion matrix
confusion_matrix=np.append(confusion_matrix_simple,[np.sum(confusion_matrix_simple,axis=0)],axis=0) # append  a row with the sum on columns
col=np.array([np.sum(confusion_matrix,axis=1)]) # sum on rows
confusion_matrix=np.append(confusion_matrix, col.T ,axis=1) # append  a column with the sum on rowss
print(">>  Confusion matrix: \n")
print(confusion_matrix)

fig, ax = plt.subplots()
im = ax.imshow(confusion_matrix_simple, cmap='plasma')
ax.set_xticks(np.arange(2))
ax.set_yticks(np.arange(2))
ax.set_ylabel('Truth', fontsize = 14)
ax.set_xlabel('Predictions', fontsize = 14)
fig.set_size_inches(4, 4, forward=True)
# plt.grid()
cbar = ax.figure.colorbar(im, ax=ax)
ax.set_title('Confusion matrix', fontsize = 18)
ax.text(-0.1, 0.1,confusion_matrix_simple[0,0], fontsize = 15, fontweight='bold')
ax.text( 0.9, 0.1,confusion_matrix_simple[0,1], fontsize = 15, fontweight='bold', color='red')
ax.text(-0.1, 1.1,confusion_matrix_simple[1,0], fontsize = 15, fontweight='bold', color='red')
ax.text( 0.9, 1.1,confusion_matrix_simple[1,1], fontsize = 15, fontweight='bold')
fig.tight_layout()
fig.savefig('confusion-matrix.png', dpi=100)
plt.show()