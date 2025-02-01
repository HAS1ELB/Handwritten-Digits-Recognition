from data_loading import load_data
from tensorflow.keras.models import load_model

# Charger les données
(x_train, y_train), (x_test, y_test) = load_data()

# Charger le modèle entraîné
model = load_model('models/mnist_model.h5')

# Évaluer le modèle
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')