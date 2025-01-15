import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_rows = 28
img_cols = 28
channels = 1

# MNIST image size.
img_shape = (channels, img_rows, img_cols)

# Noise vector size of generator.
z_dim = 100


# Used to draw an image.
def sample_images(generator, image_grid_rows=4, image_grid_columns=4):
    z = torch.randn(image_grid_rows * image_grid_columns, z_dim, device=device)
    gen_imgs = generator(z)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(4, 4), sharey=True, sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, 0].cpu().detach().numpy(), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256 * 7 * 7),
            nn.ReLU(),
            # In order to input the vector into the CNN, a shape transformation is required.
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # Notice the activation function here
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            # This is binary
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# Build generators and discriminators.
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
loss_function = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters())
optimizer_D = optim.Adam(discriminator.parameters())

losses = []
accuracies = []
iteration_checkpoints = []


def train(iterations, batch_size, sample_interval):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for iteration in range(iterations):
        for i, (real_imgs, _) in enumerate(train_loader):
            # All real data is labeled 1.
            real_label = torch.ones(real_imgs.shape[0], 1, device=device)
            # All generated data is labeled 0.
            fake_label = torch.zeros(real_imgs.shape[0], 1, device=device)
            real_imgs = real_imgs.to(device)
            real_imgs = real_imgs.to(device)
            optimizer_D.zero_grad()
            real_output = discriminator(real_imgs)
            real_loss = loss_function(real_output, real_label)
            z = torch.randn(real_imgs.shape[0], z_dim, device=device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            fake_loss = loss_function(fake_output, fake_label)
            #  Training discriminator.
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            z = torch.randn(real_imgs.shape[0], z_dim, device=device)
            gen_imgs = generator(z)
            gen_output = discriminator(gen_imgs)
            #  Training generator.
            g_loss = loss_function(gen_output, real_label)
            g_loss.backward()
            optimizer_G.step()
        if (iteration + 1) % sample_interval == 0:
            losses.append((d_loss.item(), g_loss.item()))
            accuracies.append(100.0 * (real_output.mean().item() + 1 - fake_output.mean().item()) / 2)
            iteration_checkpoints.append(iteration + 1)
            print("%d [D loss: %.4f, acc.: %.2f%%] [G loss: %.4f]" % (
                iteration + 1, d_loss.item(), 100.0 * (real_output.mean().item() + 1 - fake_output.mean().item()) / 2,
                g_loss.item()))
            sample_images(generator)


train(10000, 128, 1000)
losses = np.array(losses)

# Draw the cost function of the generator and discriminator.
plt.figure(figsize=(15, 5))
plt.plot(iteration_checkpoints, losses.T[1], label="Generator loss")
plt.plot(iteration_checkpoints, losses.T[0], label="Discriminator loss")
plt.xticks(iteration_checkpoints, rotation=90)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()
