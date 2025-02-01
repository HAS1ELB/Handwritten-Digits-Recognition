import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_loading import load_data

# Charger le modèle entraîné
model = load_model('models/mnist_model.h5')

# Charger les données
(x_train, y_train), (x_test, y_test) = load_data()

# Sélectionner une image de test (exemple : première image du jeu de test)
sample_image = x_test[0]
sample_label = y_test[0]

# Faire une prédiction
prediction = model.predict(np.expand_dims(sample_image, axis=0))
predicted_label = np.argmax(prediction)

# Afficher l'image et la prédiction
plt.imshow(sample_image.squeeze(), cmap='gray')
plt.title(f'True: {sample_label}, Predicted: {predicted_label}')
plt.show()