import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.optim as optim


class Discriminator(nn.Module):
  def __init__(self, image_dim, image_channels, channels):
    super().__init__()
    self.image_dim = image_dim
    self.image_channels = image_channels
    self.channels = channels

    # layers
    self.num_interm_layers = int(np.log2(image_dim / 4))
    for i in range(self.num_interm_layers):
      channels_in = image_channels if i == 0 else channels * 2**(i-1)
      channels_out = channels * 2**i
      setattr(self, 'conv' + str(i+1), nn.Conv2d(
        channels_in, channels_out,
        kernel_size=4, stride=2, padding=1
      ))
    
    setattr(self, 'conv' + str(self.num_interm_layers + 1), nn.Conv2d(
      channels * 2**(self.num_interm_layers - 1), channels * 2**self.num_interm_layers,
      kernel_size=5, stride=1, padding=2,
    ))
    self.fc = nn.Linear(4 * 4 * channels * 2**self.num_interm_layers, 1)

  def forward(self, x):
    for i in range(1, self.num_interm_layers + 2):
      x = F.leaky_relu(getattr(self, 'conv' + str(i))(x))
    x = x.view(-1, 4 * 4 * self.channels * 2**self.num_interm_layers)
    return self.fc(x)

class Generator(nn.Module):
  def __init__(self, noise_dim, image_dim, image_channels, channels):
    # configurations
    super().__init__()
    self.noise_dim = noise_dim
    self.image_dim = image_dim
    self.image_channels = image_channels
    self.channels = channels
    self.num_interm_layers = int(np.log2(image_dim / 4))

    # layers
    self.fc = nn.Linear(noise_dim, 4**2 * self.channels * 2**self.num_interm_layers)
    self.bn0 = nn.BatchNorm2d(channels * 2**self.num_interm_layers)

    for i in range(self.num_interm_layers):
      channels_in = channels * 2**(self.num_interm_layers - i)
      channels_out = channels * 2**(self.num_interm_layers - i - 1)
      setattr(self, 'deconv' + str(i+1), nn.ConvTranspose2d(
        channels_in, channels_out,
        kernel_size=4, stride=2, padding=1
      ))
      setattr(self, 'bn' + str(i+1), nn.BatchNorm2d(channels_out))
    
    setattr(self, 'deconv' + str(self.num_interm_layers + 1), nn.ConvTranspose2d(
      channels, image_channels,
      kernel_size=3, stride=1, padding=1
    ))

  def forward(self, z):
    g = F.relu(self.bn0(
      self.fc(z).view(-1, self.channels * 2**self.num_interm_layers, 4, 4)
    ))
    for i in range(1, self.num_interm_layers + 1):
      g = F.relu(getattr(self, 'bn' + str(i))(
        getattr(self, 'deconv' + str(i))(g)
      ))
    g = getattr(self, 'deconv' + str(self.num_interm_layers + 1))(g)
    return F.sigmoid(g)


class WGAN_GP(nn.Module):
  def __init__(self, dtype, noise_dim, image_dim, image_channels,
               disc_channels, gen_channels, dataset_name, cuda, tag):
    # Configuration values
    super().__init__()
    self.datatype = dtype
    self.noise_dim = noise_dim
    self.image_dim = image_dim
    self.image_channels = image_channels
    self.disc_channels = disc_channels
    self.gen_channels = gen_channels
    self.dataset_name = dataset_name
    self.cuda = cuda
    self.tag = tag

    # Model Components
    self.discriminator = Discriminator(
      image_dim=self.image_dim,
      image_channels=self.image_channels,
      channels=self.disc_channels,
    ).type(self.datatype)

    self.generator = Generator(
      noise_dim=self.noise_dim,
      image_dim=self.image_dim,
      image_channels=self.image_channels,
      channels=self.gen_channels,
    ).type(self.datatype)
   
  @property
  def name(self):
    return (
        'WGAN-GP'
        '-{tag}'
        '-n{noise_dim}'
        '-d{disc_channels}'
        '-g{gen_channels}'
        '-{dataset_name}-{image_dim}x{image_dim}x{image_channels}'
    ).format(
        tag=self.tag,
        noise_dim=self.noise_dim,
        disc_channels=self.disc_channels,
        gen_channels=self.gen_channels,
        image_dim=self.image_dim,
        image_channels=self.image_channels,
        dataset_name=self.dataset_name
    )

  def disc_loss(self, real_images, noise, return_g=False):
    fake_images = self.generator(noise)
    logits_real = self.discriminator(real_images)
    logits_fake = self.discriminator(fake_images)
    loss = -(logits_real.mean()-logits_fake.mean())
    return (loss, fake_images) if return_g else loss

  def gen_loss(self, noise, return_g=False):
    fake_images = self.generator(noise)
    logits_fake = self.discriminator(fake_images)
    loss = -logits_fake.mean()
    return (loss, fake_images) if return_g else loss

  def disc_optimizer(self, lr, beta1, beta2, weight_decay):
    return optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

  def gen_optimizer(self, lr, beta1, beta2, weight_decay):
    return optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

  def sample_images(self, num_images):
      return self.generator(self.sample_noise(num_images))

  def sample_noise(self, batch_size):
      return torch.randn(batch_size, self.noise_dim).type(self.datatype) * .1

  def preprocess_data(self, data):
    # return 2 * data - 1
    return data

  def gradient_penalty(self, x, g, lambda_val):
    assert x.size() == g.size()
    a = torch.rand(x.size(0), 1).type(self.datatype)
    a = a\
        .expand(x.size(0), x.nelement()//x.size(0))\
        .contiguous()\
        .view(
            x.size(0),
            self.image_channels,
            self.image_dim,
            self.image_dim
        )
    interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True).type(self.datatype)
    c = self.discriminator(interpolated)
    gradients = autograd.grad(
        c, interpolated, grad_outputs=(
            torch.ones(c.size()).type(self.datatype)
        ),
        create_graph=True,
        retain_graph=True,
    )[0]
    return lambda_val * ((1-(gradients+1e-16).norm(2, dim=1))**2).mean()
