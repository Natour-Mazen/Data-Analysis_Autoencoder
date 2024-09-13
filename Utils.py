from matplotlib import pyplot as plt


def afficher_images(n, x_test, decoded_imgs):
    """
    Cette fonction affiche les images originales et leurs reconstructions.

    Paramètres : \n
    * n (int) : nombre d'images à afficher \n
    * x_test (array) : images originales \n
    * decoded_imgs (array) : images reconstruites
    """
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # affichage des images originales
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Image Originale')

        # affichage des images reconstruites
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('Image Reconstruite')
    plt.suptitle('Comparaison des Images Originales et Reconstruites')
    plt.show()


def afficher_courbes_perte(autoencoder_train, nepochs):
    """
    Cette fonction affiche les courbes de perte d'entraînement et de validation.

    Paramètres :\n
    * autoencoder_train (History) : historique d'entraînement de l'autoencodeur \n
    * nepochs (int) : nombre d'époques
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(nepochs), autoencoder_train.history['loss'], label='Perte d\'entraînement')
    plt.plot(range(nepochs), autoencoder_train.history['val_loss'], label='Perte de validation')
    plt.title('Pertes d\'entraînement et de validation au fil des époques')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.show()
