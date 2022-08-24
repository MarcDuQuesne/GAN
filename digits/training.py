
from digits.models import GAN, Discriminator, Generator

plt.axis('off')
plt.tight_layout()

def training(epochs=1, batch_size=128):

    for e in range(1,epochs+1 ):
        #print("Epoch %d" %e)
        #tqdm()
        for _ in range(batch_size):
        #generate  random noise as an input  to  initialize the  generator


        if e == 1 or e % 20 == 0:

            plot_generated_images(gan.generator)

training(400,128)




gan = GAN(discriminator=Discriminator(), generator=Generator)