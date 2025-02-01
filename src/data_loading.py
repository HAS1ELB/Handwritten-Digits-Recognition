import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    # Charger les données MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normaliser les images entre 0 et 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Redimensionner pour les modèles CNN (ajouter une dimension pour le canal)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    return (x_train, y_train), (x_test, y_test)