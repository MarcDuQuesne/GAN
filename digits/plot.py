import matplotlib.pyplot as plt
import numpy as np

def plot_generated_images(generator, examples=100, dim=(10,10), figsize=(10,10)):

    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100,28,28)
    plot_images(generated_images, dim, figsize)


def plot_images(images, dim=(10,10), figsize=(10,10)):

    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i],
                   interpolation = 'nearest',
                   cmap = 'gray')
        plt.axis('off')
    plt.tight_layout()