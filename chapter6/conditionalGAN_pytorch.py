import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

img_rows = 28
img_cols = 28
channels = 1
num_classes = 10

img_shape = (channels, img_rows, img_cols)

z_dim = 100


def sample_images(image_grid_rows=2, image_grid_columns=5):
    z = torch.randn(image_grid_rows * image_grid_columns, z_dim)
    labels = torch.arange(0, 10)
    gen_imgs = generator(z, labels)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10, 4), sharey=True, sharex=True)
    cnt = 0
    for i in range(image_grid_rows):
        for j in range(image_grid_columns):
            axs[i, j].imshow(gen_imgs[cnt, 0, :, :].detach().cpu(), cmap='gray')
            axs[i, j].axis('off')
            axs[i, j].set_title("Digit: %d" % labels[cnt])
            cnt += 1


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, z_dim)
        self.model = nn.Sequential(
            nn.Linear(2 * z_dim, 256 * 7 * 7),  # 128*256 * 7*7
            nn.Flatten(),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z, label):
        gen_input = torch.cat((self.label_emb(label), z), dim=1)
        gen_input = gen_input.view(gen_input.size(0), -1, 1, 200)
        gen_output = self.model(gen_input)
        return gen_output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, channels)
        self.model = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, label):
        labels_emb = self.label_emb(label).unsqueeze(2).unsqueeze(3).expand(-1, -1, img.size(2), img.size(3))
        disc_input = torch.cat((img, labels_emb), dim=1)
        disc_output = self.model(disc_input)
        return disc_output.view(-1, 1).squeeze(1)


generator = Generator(z_dim)
discriminator = Discriminator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

d_optimizer = optim.Adam(discriminator.parameters())
g_optimizer = optim.Adam(generator.parameters())
criterion = nn.BCELoss()


def train(iterations, batch_size, sample_interval):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for iteration in range(iterations):
        for i, (imgs, labels) in enumerate(train_loader):
            real_label = torch.ones(imgs.shape[0], 1, device=device)
            fake_label = torch.zeros(imgs.shape[0], 1, device=device)
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Training discriminator.
            discriminator.zero_grad()
            real_validity = discriminator(imgs, labels)
            real_loss = criterion(real_validity, real_label.view(-1))

            z = torch.randn(imgs.shape[0], z_dim).to(device)

            fake_imgs = generator(z, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)

            fake_loss = criterion(fake_validity, fake_label.view(-1))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Training generator.
            generator.zero_grad()
            z = torch.randn(imgs.shape[0], z_dim).to(device)
            fake_imgs = generator(z, labels)
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_validity, real_label.view(-1))

            g_loss.backward()
            g_optimizer.step()

            if (iteration + 1) % sample_interval == 0:
                print("[Iteration %d/%d] [D loss: %f] [G loss: %f]" % (
                    iteration + 1, iterations, d_loss.item(), g_loss.item()))

        if (iteration + 1) % sample_interval == 0:
            sample_images()


train(10000, 128, 1000)

image_grid_rows = 10
image_grid_columns = 5
z = torch.randn(image_grid_rows * image_grid_columns, z_dim).to(device)
labels_to_generate = torch.tensor([[i for j in range(5)] for i in range(10)]).flatten().to(device)
gen_imgs = generator(z, labels_to_generate)
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(image_grid_rows, image_grid_columns, figsize=(10, 20), sharey=True, sharex=True)
cnt = 0
for i in range(image_grid_rows):
    for j in range(image_grid_columns):
        axs[i, j].imshow(gen_imgs[cnt, 0, :, :].detach().cpu(), cmap='gray')
        axs[i, j].axis('off')
        axs[i, j].set_title("Digit: %d" % labels_to_generate[cnt])
        cnt += 1
plt.show()
