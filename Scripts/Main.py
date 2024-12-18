import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #stop the tensorflow warnings
from keras.models import Sequential
from keras.layers import Dense

#
# url = 'https://drive.google.com/uc?export=download&id=1MOZaQ5vmUu6ChdfTva-qjZ__UAiOmhh9'  #1461 house inquiries
# output_file = 'housepricedata.csv'
#
# response = requests.get(url)
# if response.status_code == 200:
#     with open(output_file, 'wb') as file:
#         file.write(response.content)
#     print(f"File downloaded as {output_file}")
# else:
#     print(f"Failed to download file. Status code: {response.status_code}")
#

DF = pd.read_csv('housepricedata.csv')
#print(DF)

DataFrame = DF.values #converting the data into arrays


X = DataFrame[:, 0:10]  # input data set
Y = DataFrame[:, 10] # output data


min_max_scaler = preprocessing.MinMaxScaler()
X_sc = min_max_scaler.fit_transform(X) # scales the dataset so it is between 0 and 1



X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_sc, Y, test_size=0.3)
# val_and_test data will be 30%, train set will be 70%

X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.6)
# - test set 18% (60% * 30% = 18%)
# - validation set 12%

model = Sequential([
                    Dense(16, activation='relu', input_shape=(10,), name = 'HiddenLayer1'), #10 input features
                    Dense(8, activation='relu', name = 'HiddenLayer2'),
                    Dense(1, activation='sigmoid', name = 'OutputLayer'),
])

print("The generated Sequential model is:")
model.summary()
print("\nCurrent Working Directory where the model will be saved is:" , os.getcwd())



#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(
    X_train,
    Y_train,
    batch_size = 2,  # number of samples per gradient update (default 32), size of mini-batch.
    epochs=100,  #  number of epochs to train the model

    verbose = 1, # 1 for progress bar, 2 for each row for an epoch
    validation_data=(X_val, Y_val))
#improved accuracy to 0.91 32 mini batch and 300 epochs
#using the adam optimizer still works best by far
#max accuracy obtained is still 0.92



