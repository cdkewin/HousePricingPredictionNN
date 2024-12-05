import Main


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
fig.suptitle('Loss and accuracy during training')

ax1.plot(Main.hist.history['loss']) # extract the loss values from the history object generated in the training phase
ax1.plot(Main.hist.history['val_loss'])
ax1.set_title('Model loss during training')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper right')

ax2.plot(Main.hist.history['accuracy'])
ax2.plot(Main.hist.history['val_accuracy'])
ax2.set_title('Model accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

test_loss, test_accuracy = Main.model.evaluate(
    Main.X_test,
    Main.Y_test,
    verbose = 2)
print('\ntest_loss = ', test_loss)
print('test_accuracy = ', test_accuracy)