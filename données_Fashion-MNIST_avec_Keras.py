import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
from Utils import afficher_courbes_perte, afficher_images

print("\n--- 1 ----\n")

# Charger le jeu de données Fashion-MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Vérifier les dimensions des données
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Affichage des 5 premieres images brut
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i])
    ax.set_title(f"Image {i + 1}")
    ax.grid(False)
plt.suptitle("5 premières images de l'ensemble de données Fashion-MNIST")
plt.show()

# Normalisation des images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Vectorisation des images 28x28 en vecteur de 784 coeffs
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("x_train.shape: ")
print(x_train.shape)
print("x_test.shape: ")
print(x_test.shape)

print("\n--- 2 ----\n")

print("\n--- 2.1 ----\n")

# Taille du code dans l'espace latent
taille_code = 32

# Images d'entrée
input_img = keras.Input(shape=(784,))

# Encodeur
encoded = layers.Dense(taille_code, activation='relu')(input_img)

# Décodeur
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# Modèle pour l'auto encodeur
autoencoder = keras.Model(input_img, decoded)

# Modèle d'encodeur sur les données d'entrée pour obtenir le code
encoder = keras.Model(input_img, encoded)

# Entrée du décodeur : code dans l'espace latent
encoded_input = keras.Input(shape=(taille_code,))

# Récupérer la dernière couche de l'auto encodeur
decoder_layer = autoencoder.layers[-1]

# Modèle du décodeur
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# Visualiser et analyser la configuration du réseau de neurones
autoencoder.summary()

print("\n--- 2.2 ----\n")

# Entraîner le modèle
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
nepochs = 20
history = autoencoder.fit(x_train, x_train,
                          epochs=nepochs,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test))

# Affichage des courbes de perte
afficher_courbes_perte(history, nepochs)

print("\n--- 2.3 ----\n")

# Encodage puis décodage des images de test
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Nombre d'images à afficher
n = 5
afficher_images(n, x_test, decoded_imgs)

# Tester le modèle
decoded_imgs = autoencoder.predict(x_test)

# Calculer le SSIM et le MSE
ssim_values = [ssim(x_test[i], decoded_imgs[i], data_range=1.0) for i in range(len(x_test))]
mse_values = [mse(x_test[i], decoded_imgs[i]) for i in range(len(x_test))]

# Tracer les boîtes à moustaches
plt.figure(figsize=(12, 6))

# Tracer les valeurs de SSIM
plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, sous-figure 1
plt.boxplot(ssim_values)
plt.title('Valeurs de SSIM')

# Tracer les valeurs de MSE
plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, sous-figure 2
plt.boxplot(mse_values)
plt.title('Valeurs de MSE')

plt.show()

print("\n--- 2.4 ----\n")

# Définir différentes valeurs pour le nombre d'époques
epochs_values = [10, 20, 30, 40, 50]

# Stocker les résultats des expériences
ssim_results_epochs = []

# Boucle sur différentes valeurs d'époques
for nepochs in epochs_values:
    # Construire et entraîner l'autoencodeur avec le nombre d'époques actuel
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder_train = autoencoder.fit(x_train, x_train,
                                        epochs=nepochs,
                                        batch_size=256,
                                        shuffle=True,
                                        validation_data=(x_test, x_test))
    # Prédire et évaluer
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # Calculer et stocker les résultats SSIM
    ssim_values = [ssim(x_test[i], decoded_imgs[i], data_range=1.0) for i in range(5)]
    ssim_results_epochs.append(ssim_values)

print("\nssim_results_epochs :\n")
print(ssim_results_epochs)
print("\n")
# Définir différentes valeurs pour la taille du code
code_sizes = [16, 32, 64, 256, 1024]

# Stocker les résultats des expériences
ssim_results_code_size = []

# Boucle sur différentes valeurs de taille de code
for taille_code in code_sizes:
    # Construire et entraîner l'autoencodeur avec la taille de code actuelle
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder_train = autoencoder.fit(x_train, x_train,
                                        epochs=20,
                                        batch_size=256,
                                        shuffle=True,
                                        validation_data=(x_test, x_test))

    # Prédire et évaluer
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    # Calculer et stocker les résultats SSIM
    ssim_values = [ssim(x_test[i], decoded_imgs[i], data_range=1.0) for i in range(5)]
    ssim_results_code_size.append(ssim_values)


print("ssim_results_code_size :\n")
print(ssim_results_code_size)
print("\n")

print("\n--- 3 ----\n")
print("\n--- 3.1 ----\n")


# Définition de l'entrée de notre autoencodeur
input_img = keras.Input(shape=(784,))

# "encoded" est la représentation codée de notre entrée
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

# "decoded" est la reconstruction lossy de l'entrée
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# Ceci modèle notre autoencodeur
autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

print("--- 3.2 ----")

# Compilation de l'autoencodeur
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entraînement de l'autoencodeur
nepochs = 50
autoencoder_train = autoencoder.fit(x_train, x_train,
                                    epochs=nepochs,
                                    batch_size=256,
                                    shuffle=True,
                                    validation_data=(x_test, x_test))

print("\n--- 3.3 ----\n")

# Prédiction sur l'ensemble de test
decoded_imgs = autoencoder.predict(x_test)

# Affichage des images originales et reconstruites
n = 5  # nombre d'images à afficher
afficher_images(n, x_test, decoded_imgs)

# Affichage des courbes de perte d'entraînement et de validation
afficher_courbes_perte(autoencoder_train, nepochs)
