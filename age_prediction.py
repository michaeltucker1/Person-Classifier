import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, InputLayer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load CSV data
df = pd.read_csv("data.csv")

# Data insights
minimum_value = df['age'].min()
maximum_value = df['age'].max()

print("----- Data Insights -----")
print("Minimum value:", minimum_value)
print("Maximum value:", maximum_value)
print("-------------------------")



# Preprocess dataframe
df.dropna(subset=['age'], inplace=True)
df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))

X = np.array(df['pixels'].to_list())
X = X.reshape(-1, 48, 48, 1)

# Temp - View image from any index
# index = 5005
# plt.xlabel(
#         "Age:"+str(df['age'].iloc[index])+
#         "  Ethnicity:"+str(df['ethnicity'].iloc[index])+
#         "  Gender:"+ str(df['gender'].iloc[index])
#     )
# plt.imshow(X[index])
# plt.show()

df["age"] = pd.cut(df["age"],bins=[0, 5, 10, 20, 40, 60, 80, 100, 120], labels=["0","1","2","3","4","5","6","7"]) 

y = to_categorical(np.array(df['age']),num_classes=8)

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=42)


model = Sequential()
model.add(InputLayer(shape=(48, 48, 1)))
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.01))) 
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',kernel_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(256,activation='relu',kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(8,activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=10, min_delta=0.001,restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

epochs = 40
batch_size = 64

history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test), callbacks=[early_stopping, learning_rate_reduction])

model.save("model/age_prediction_model.keras")