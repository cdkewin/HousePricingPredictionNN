import Main

x = Main.X_test[:5, :]  # the first 5 examples in the test dataset
yhat = Main.model(x)
print('Input sample: \n\n', x)
print('\nPredicted outputs: \n\n', yhat.numpy())



ns = 16 # number of samples

print('Make predictions for the first ' + str(ns) + ' samples in the independent test dataset')
predictions = Main.model.predict(Main.X_test[:ns]) # generate predictions for input samples
pred_binary = np.where(predictions > 0.5, 1, 0) # convert to binary prediction, treshold=0.5
print("\nPredictions:\n", predictions.reshape(1,ns))
print("\nBinary prediction:\n", pred_binary.reshape(1,ns))
print("\nTargets:\n", Main.Y_test[:ns])


# plot predictions vs targets

xx = np.linspace(0, ns, ns)
fig = plt.gcf()
fig.set_size_inches(20, 5)
plt.plot([0, ns], [0.5, 0.5], color='red')
for idx in xx:
  plt.axvline(x=idx, ymin=0, ymax=1, color='black', linestyle=':', linewidth = 0.7)

plt.scatter(xx, Main.Y_test[:ns], linewidths = 3, label = 'target')
plt.scatter(xx, predictions, linewidths = 3, label = 'predicted')
plt.title('Predicted vs targets', color = 'magenta', fontsize=15)
plt.ylabel('predicted / target', color = 'magenta', fontsize=13)
plt.xlabel('data index', color = 'magenta', fontsize=13)
plt.legend(loc='center right', fontsize='x-large')
fig.savefig('predictions_vs_target.png', dpi=100)
plt.show()

fig = plt.gcf()
fig.set_size_inches(20, 3)
for idx in xx:
  plt.axvline(x=idx, ymin=0, ymax=1, color='black', linestyle=':', linewidth = 0.7) # vertical line

plt.scatter(xx, Main.Y_test[:ns], marker='D', linewidths = 5, label = 'target')
plt.scatter(xx, pred_binary, marker='.' ,linewidths = 3, label = 'predicted')
plt.title('Binary predicted vs targets', color = 'magenta', fontsize=15)
plt.ylabel('binary predicted / target', color = 'magenta', fontsize=13)
plt.xlabel('data index', color = 'magenta', fontsize=13)
plt.legend(loc='center right', fontsize='x-large')
fig.savefig('predictions_binary_vs_target.png', dpi=100)
plt.show()

test_loss, test_accuracy = Main.model.evaluate(
    Main.X_test[0:ns],
    Main.Y_test[0:ns],
    verbose = 0)
print('\n\ntest_accuracy = ', test_accuracy)
diffs = np.subtract(pred_binary.reshape(1,ns), Main.Y_test[0:ns])
num_diffs = np.sum(abs(diffs), axis = 1)
print ('\nNumber of wrong predictions: ', *num_diffs, 'out of ', ns, 'examples, (', *(num_diffs/ns*100),'%)')
