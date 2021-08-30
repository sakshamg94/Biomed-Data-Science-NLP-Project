import tensorflow as tf
from keras.layers import Dense, Embedding, Bidirectional, LSTM
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle 
import numpy as np
from sklearn.metrics import confusion_matrix


# VARIABLES USED FOR MODEL SAVING, EVALUATING AND PLOTTING
training_time=1
model_num = 1 # which modekl? Could be a string: used for file save name
label2className = {} # dictionary to map numerical categories to their names : eg {1:"injury", 2:"endocrine"}

# DATA PROCESSING
# RESHAPE INPUT TO THE DESIRED FORM FOR LSTM TRAINING:
# NEED 3D tensors of shape: [n_samples, n_timesteps = 2, n_features = input vector dim]
train_X = None
train_y = None
val_X  = None
val_y = None
test_X = None
test_y =None



# BUILD THE MODEL BY ADDING LAYERS
model = tf.keras.models.Sequential()
# the following is a placeholder for where I imagine our embedding layer will go
# nevertheless note that keras 
model.add(tf.keras.layers.Embedding(input_dim = INPUT_VEC_SIZE, output_dim= EMBED_SIZE, input_length = MAX_INPUT_VEC_LENGTH))

# add a bidirectional LSTM layer to not only take the past context of a vector into account but also the future context
# can play with the activation type here: tanh(default) or relu being popular options
# num_hidden_units
model.add(Bidirectional(LSTM(units = NUM_LSTM_UNITS, input_shape= (train_X.shape[1], train_X.shape[2]))))

# we can add another Bidirectional LSTM layer if need be -- but we will need to evaluate that later -- no need rn
# model.add(Bidirectional(LSTM(num_hidden_units, input_shape= (train_X.shape[1], train_X.shape[2]))))

# add dense units
model.add(Dense(units = NUM_CLASSES, activation='softmax'))
model.summary()


# COMILE THE MODEL
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=0.001), metrics=['acc'])

# FIT THE NETWORK
hist = model.fit(train_X, train_y, epochs= EPOCHS, batch_size = BATCH_SIZE, validation_data = (val_X, val_y))

# save the model and train histry
if training_time == 1:
    # save the model
    model.save('model_{}.h5'.format(model_num))  # creates a HDF5 file 
    with open('model_{}.pickle'.format(model_num), 'wb') as file_pi:
        pickle.dump(hist, file_pi)
elif training_time == 0:
    # loading the saved model
    model = load_model('model_{}.h5'.format(model_num))
    # Now run model.compile from before if you need to 
    with open('model_{}.pickle'.format(model_num), 'rb') as file_pi:
        hist = pickle.load(file_pi)        


# PLOT HISTORY
plt.plot(hist.history['loss'], label = 'train')
plt.plot(hist.history['val_loss'], label = 'validation')
plt.legend()
plt.show()

plt.plot(hist.history['acc'], label = 'train_acc')
plt.plot(hist.history['val_acc'], label = 'validation_acc')
plt.legend()
plt.show()

# PRINT THE STATS
avg_train_acc = np.average(his.history['acc'][10:])*100
avg_valid_acc = np.average(his.history['val_acc'][10:])*100
print("Average training acc: ", avg_train_acc)
print("Average validation acc: ", avg_valid_acc)

# EVALUATE THE MODEL
# make predictions
test_ypred = model.predict(test_X)

# VISUALIZE PREDICTIVE CAPACITY USING CONFUSION MATRICES
cm = confusion_matrix(test_y, test_ypred)
np.set_printoptions(precision=2)
plt.figure()

class_names = []
    for k,v in label2className.items():
        class_names.append(k)

plot_confusion_matrix(cm, class_names)
plt.savefig('Confusion_matrix_model_{}.png'.format(model_num))
plt.show()

# WE WILL ALSO WANT TO LOK AT THE CLASSIFICATION REPORT FROM ASSIGNMENT 2  
# -- HOW TO EXTEND THAT TO MULTICLASS CLASSIFICATION
