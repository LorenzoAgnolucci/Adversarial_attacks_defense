import torch.nn as nn
import torchvision.models as models
import torch
import torch_model as leogan
from jpegLayer import jpegLayer
import numpy as np


class JpegComp(nn.Module):
    def __init__(self):
        super(JpegComp, self).__init__()

    def forward(self, input_, qf):
        return jpegLayer(input_, qf)


class Normalize(nn.Module):
    # class for normalizing as a NN module
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


class ComposedModel(nn.Module):
    def __init__(self, arch='resnet50', qf=20, defence='gan', jpeg_pass=1, delta_qf=0, model_iterations=1,
                 multi_gan=False):
        super(ComposedModel, self).__init__()
        assert defence in ['gan', 'jpeg', None]
        self.qf = qf
        self.delta_qf = delta_qf
        self.defence = defence
        self.model_iterations = model_iterations
        self.multi_gan = multi_gan
        if defence is not None:
            self.jpeg = JpegComp()
            self.jpeg_pass = jpeg_pass
        if defence == 'gan':
            # Pre and post normalization for gan
            self.preGan = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.postGan = Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
            # Load gan
            self.gan = leogan.SRGAN_g()
            if self.multi_gan:
                # Fixed models to QF 20, 40, 60. Change in the future
                self.gan_params = []
                self.available_qf = (20, 40, 60)
                print('Loading multiple GAN models')
                for q in self.available_qf:
                    self.gan_params.append(torch.load('gan_models_pt/torch_' + str(q) + '.pt'))
            else:
                self.gan.load_state_dict(torch.load('gan_models_pt/torch_' + str(self.qf) + '.pt'))
        # Load main net
        if arch == 'resnet50':
            self.cnn = models.resnet50(pretrained=True)
        elif arch == 'mobilenet':
            self.cnn = models.mobilenet_v2(pretrained=True)
        elif arch == 'inception':
            self.cnn = models.inception_v3(pretrained=True)
        elif arch == 'densenet':
            self.cnn = models.densenet121(pretrained=True)
        else:
            print('Unsupported architecture')
        # Normalization for PT model
        self.prePT = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        # PREVEDERE QF IN INPUT AL FORWARD DELLA RETE MAIN. Va cambiato quindi anche il costruttore e la chiamata
        # dentro l'attacco. Si puo fare volendo con una fun ausiliaria?
        for _ in range(self.model_iterations):
            if self.defence is not None:
                for _ in range(self.jpeg_pass):
                    curr_qf = int(self.qf + np.random.randint(low=-self.delta_qf, high=self.delta_qf+1, size=(1,)))
                    print('Current QF: ', curr_qf)
                    x = self.jpeg(x, curr_qf)
            if self.defence == 'gan':
                # Change  GAN weights at each iteration based on closest QF
                if self.multi_gan:
                    nearest = min(self.available_qf, key=lambda v: abs(curr_qf - v))
                    self.gan.load_state_dict(self.gan_params[self.available_qf.index(nearest)])

                # Pre process in [-1,1]
                x = self.preGan(x)
                # Restore with gan
                x = self.gan(x)
                # Map back to [0,1]
                x = self.postGan(x)
        # Preprocess for network i.e. normalize mean std
        x = self.prePT(x)
        # Classify
        x = self.cnn(x)

        return x