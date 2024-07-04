# High-Performance Computing and Big Data: Deep Learning Model Training

This repository contains code for training a deep learning model using TensorFlow and Keras. The model is trained on a dataset to classify data into multiple categories. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Visualization](#visualization)
7. [Usage](#usage)

## Project Overview

The project focuses on building a Convolutional Neural Network (CNN) for classification tasks. It includes:
- Data loading and preprocessing
- Model building using Keras
- Training with callbacks
- Evaluation and metrics
- Visualization of accuracy, loss, confusion matrix, and ROC AUC curves

## Requirements

The following libraries are required to run the code:
- numpy
- tensorflow
- seaborn
- pandas
- matplotlib
- scikit-learn
- imbalanced-learn

To install the required libraries, run:
```bash
pip install numpy tensorflow seaborn pandas matplotlib scikit-learn imbalanced-learn

Data Preparation
The dataset is loaded from a CSV file and split into training and testing sets. The code includes options for data augmentation using SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

dataset1 = np.loadtxt('/path/to/dataset.csv', delimiter=",")
X = dataset1[:, 0:10]
Y = dataset1[:, 10]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
Y_train = to_categorical(y_train - 1, classifications)
Y_test = to_categorical(y_test - 1, classifications)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

Model Training
A Convolutional Neural Network (CNN) is defined using Keras Sequential API. The model is compiled and trained with various callbacks for monitoring and logging.
model = Sequential()
# Add layers to the model
model.add(Convolution1D(filters=32, kernel_size=3, activation='relu', input_shape=(dimension, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classifications, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

time_callback = TimeHistory()
history = model.fit(X_train, Y_train, epochs=epochs, verbose=1, batch_size=32,
                    validation_data=(X_test, Y_test),
                    callbacks=[time_callback])

Evaluation
The models performance is evaluated using metrics such as accuracy, confusion matrix, and classification report.
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"\nLoss: {loss:.2f}, Accuracy: {accuracy * 100:.2f}%")

Visualization
The training history is visualized using matplotlib, including accuracy and loss plots, confusion matrix heatmap, and ROC AUC curves.

Accuracy and Loss Plots
plt.plot(history.history['accuracy'], 'b-')
plt.plot(history.history['val_accuracy'], 'r--')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig('/path/to/save/model_accuracy.png')

plt.plot(history.history['loss'], 'y-')
plt.plot(history.history['val_loss'], 'g--')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig('/path/to/save/model_loss.png')

Confusion Matrix
cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(10, 7))
sn.heatmap(cm_normalized, annot=True, cmap='Blues', cbar=False)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('/path/to/save/confusion_matrix.png')

ROC AUC Curves
fpr, tpr, roc_auc = {}, {}, {}
for i in range(classifications):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], model.predict(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(classifications):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Analysis')
plt.legend(loc="lower right")
plt.savefig('/path/to/save/roc_auc.png')

Usage
To run the code, ensure the dataset is placed in the correct directory and modify the file paths as needed. Execute the script to train the model and generate the visualizations.

bash
python train_model.py

Make sure to replace `path/to/save/model_accuracy.png`, `path/to/save/model_loss.png`, `path/to/save/confusion_matrix.png`, and `path/to/save/roc_auc.png` with the actual paths where the images will be saved.
```

![Model Accuracy](/model_accuracy.png)
![Model Loss](/model_loss.png)
![Confusion Matrix With Normalization](/confusion_matrix.png)

