from data_loading import load_data
from model import build_model

# Charger les données
(x_train, y_train), (x_test, y_test) = load_data()

# Construire le modèle
model = build_model()

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Sauvegarder le modèle
model.save('models/mnist_model.h5')