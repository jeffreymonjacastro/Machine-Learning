import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets


"""
Naive Bayes
"""

from sklearn.naive_bayes import MultinomialNB

digitos = datasets.load_digits()
sample_digit = digitos.data[:10]

fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_digit[i].reshape(8, 8), cmap="binary")
    ax.set_title(f"Digito {i}")
    ax.axis("off")

plt.show()

x = digitos.data
y = digitos.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=123)

nb = MultinomialNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues", values_format="d")

# print(metrics.classification_report(y_test, y_pred))

import numpy as np

random_indices = np.random.choice(x_test.shape[0],
                                  size=10, replace=False)
sample_images = x_test[random_indices]
true_labels = y_test[random_indices]

predicted_labels = nb.predict(sample_images)

fig, axes = plt.subplots(2, 5, figsize=(10, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i].reshape(8, 8), cmap="binary")
    ax.set_title(f"Prediccion: {predicted_labels[i]}")
    ax.axis("off")


