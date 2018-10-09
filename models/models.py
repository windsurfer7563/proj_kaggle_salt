from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from models.senet import se_resnext50_32x4d, se_resnext101_32x4d


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters=32, pretrained=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


def unet11(pretrained=False, **kwargs):
    """
    pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
            carvana - all weights are pre-trained on
                Kaggle: Carvana dataset https://www.kaggle.com/c/carvana-image-masking-challenge
    """
    model = UNet11(pretrained=pretrained, **kwargs)

    if pretrained == 'carvana':
        state = torch.load('TernausNet.pt')
        model.load_state_dict(state['model'])
    return model


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)

class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv = True):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool
                                   )

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)
            #x_out = F.sigmoid(x_out_log)

        return x_out



class SqEx(nn.Module):
    """
    Spatial Squeese and Channel Excitation
    """
    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        return y


class cSqEx(nn.Module):
    """
    Chanel Squeese and Spatial Excitation
    """
    def __init__(self, n_features):
        super(cSqEx, self).__init__()

        self.spatial_se = nn.Sequential(nn.Conv2d(n_features, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        spa_se = self.spatial_se(x)
        return spa_se



class DecoderBlockV3(nn.Module):
    """
    Decoder block with added SE blocks,
    interpolation as up-scaling
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockV3, self).__init__()
        self.in_channels = in_channels

        self.block = nn.Sequential(
            Conv3BN(in_channels, middle_channels),
            Conv3BN(middle_channels, out_channels)
            )
        #self.is_deconv = is_deconv
        self.spatial_gate = SqEx(out_channels)
        self.channel_gate = cSqEx(out_channels)

    def forward(self, x, e=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x,e], 1)
        x = self.block(x)
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 * x + g2 * x
        return x



class ResNet34(nn.Module):
    """
    Decoder ResNext34
    SE blocks in decoder
    + Hypercolumn
    """
    def __init__(self, num_classes=1, num_filters=32, pretrained=True, is_deconv = False, dropout_2d=0):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()


        self.num_classes = num_classes

        self.dropout_2d = dropout_2d

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   #self.pool
                                   )


        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4


        bottom_channel_nr = 512


        self.center = nn.Sequential(
            Conv3BN(512, 512, bn=True),
            Conv3BN(512, 256, bn=True),
            self.pool
        )


        self.dec5 = DecoderBlockV3(bottom_channel_nr + 256, 512, 64)
        self.dec4 = DecoderBlockV3(bottom_channel_nr // 2 + 64, 256, 64)
        self.dec3 = DecoderBlockV3(bottom_channel_nr // 4 + 64, 128, 64)
        self.dec2 = DecoderBlockV3(bottom_channel_nr // 8 + 64, 64, 64)
        self.dec1 = DecoderBlockV3(64, 32, 64)


        self.final = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )


    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)#;              print("conv5: ", conv5.size())


        center = self.center(conv5)#; print("center: ", center.size())

        dec5 = self.dec5(center, conv5)
        dec4 = self.dec4(dec5, conv4)
        dec3 = self.dec3(dec4, conv3)
        dec2 = self.dec2(dec3, conv2)
        dec1 = self.dec1(dec2)


        # hypercolumn
        f = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(dec3, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(dec4, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(dec5, scale_factor=16, mode="bilinear", align_corners=False),
        ), 1)
        f = F.dropout2d(f, p=0.2)

        x_out = self.final(f)

        return x_out


class TTAFunction(nn.Module):
    """
    # class with metod tta_flip
    # use it below for 'Class Inheritance'
    # only h-flip
    """
    def tta_flip(self, x):
        self.eval()
        with torch.no_grad():
            result = torch.sigmoid(self.forward(x))
            result += torch.sigmoid(self.forward(x.flip(3)).flip(3)) # apply flip and back
        return 0.5*result



class SE_ResNext50(TTAFunction):

    def __init__(self, num_classes=1):

        super().__init__()
        self.is_deconv = False
        is_deconv = False
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')


        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   )

        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        bottom_channel_nr = 2048

        self.center = nn.Sequential(
            Conv3BN(2048, 512, bn=True),
            Conv3BN(512, 256, bn=True),
            self.pool
        )

        self.dec5 = DecoderBlockV3(bottom_channel_nr + 256, 512, 64)
        self.dec4 = DecoderBlockV3(bottom_channel_nr // 2 + 64, 256, 64)
        self.dec3 = DecoderBlockV3(bottom_channel_nr // 4 + 64, 128, 64)
        self.dec2 = DecoderBlockV3(bottom_channel_nr // 8 + 64, 64, 64)
        self.dec1 = DecoderBlockV3(64, 32, 64)

        self.final = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )



    def forward(self, x):
        #print("x: ", x.size())
        conv1 = self.conv1(x)#;                print("conv1: ", conv1.size())
        conv2 = self.conv2(conv1)#;              print("conv2: ", conv2.size())
        conv3 = self.conv3(conv2)#;              print("conv3: ", conv3.size())
        conv4 = self.conv4(conv3)#;              print("conv4: ", conv4.size())
        conv5 = self.conv5(conv4)#;              print("conv5: ", conv5.size())

        center = self.center(conv5)#; print("center: ", center.size())
        dec5 = self.dec5(center, conv5)#;print("dec5: ", dec5.size())
        dec4 = self.dec4(dec5, conv4)#;print("dec4: ", dec4.size())
        dec3 = self.dec3(dec4, conv3)#;print("dec3: ", dec3.size())
        dec2 = self.dec2(dec3, conv2)#;print("dec2: ", dec2.size())
        dec1 = self.dec1(dec2)#;print("dec1: ", dec1.size())

        # hypercolumn
        f = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(dec3, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(dec4, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(dec5, scale_factor=16, mode="bilinear", align_corners=False),
        ), 1)
        f = F.dropout2d(f, p=0.4, training=self.training)
        final = self.final(f)

        return final

class SE_ResNext50_2(TTAFunction):

    def __init__(self, num_classes=1):

        super().__init__()
        self.is_deconv = False
        is_deconv = False
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')


        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   )

        #(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #(bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #(relu1): ReLU(inplace)
        #(pool): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)


        #self.conv2 = nn.Sequential(nn.MaxPool2d(kernel_size =2, stride = 2),
        #                        self.encoder.layer1
        #                          )

        self.conv2 = nn.Sequential(nn.MaxPool2d(kernel_size =2, stride = 2),
                                 self.encoder.layer1
                                   )
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        bottom_channel_nr = 2048

        self.center = nn.Sequential(
            Conv3BN(2048, 512, bn=True),
            Conv3BN(512, 256, bn=True),
            self.pool
        )

        self.dec5 = DecoderBlockV3(bottom_channel_nr + 256, 512, 64)
        self.dec4 = DecoderBlockV3(bottom_channel_nr // 2 + 64, 256, 64)
        self.dec3 = DecoderBlockV3(bottom_channel_nr // 4 + 64, 128, 64)
        self.dec2 = DecoderBlockV3(bottom_channel_nr // 8 + 64, 64, 64)
        self.dec1 = DecoderBlockV3(64 + 64, 32, 64)

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )


    def forward(self, x):
        batch_size = x.size()[0]
        #print("x: ", x.size())
        conv1 = self.conv1(x)#;                print("conv1: ", conv1.size())
        conv2 = self.conv2(conv1)#;              print("conv2: ", conv2.size())
        conv3 = self.conv3(conv2)#;              print("conv3: ", conv3.size())
        conv4 = self.conv4(conv3)#;              print("conv4: ", conv4.size())
        conv5 = self.conv5(conv4)#;              print("conv5: ", conv5.size())

        center = self.center(conv5)#; print("center: ", center.size())

        dec5 = self.dec5(center, conv5);print("dec5: ", dec5.size())
        dec4 = self.dec4(dec5, conv4);print("dec4: ", dec4.size())
        dec3 = self.dec3(dec4, conv3);print("dec3: ", dec3.size())
        dec2 = self.dec2(dec3, conv2);print("dec2: ", dec2.size())
        dec1 = self.dec1(dec2, conv1);print("dec1: ", dec1.size())

        # hypercolumn
        f = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(dec3, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(dec4, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(dec5, scale_factor=16, mode="bilinear", align_corners=False),
        ), 1)
        f = F.dropout2d(f, p=0.4, training=self.training)
        logit_pixel = self.logit_pixel(f)

        f = F.adaptive_avg_pool2d(conv5, output_size=1).view(batch_size, -1)
        f = F.dropout(f, p=0.4, training=self.training)
        logit_image = self.logit_image(f).view(-1)

        #print(logit_image.size())
        return logit_pixel, logit_image


class SE_ResNext101(TTAFunction):

    def __init__(self, num_classes=1):

        super().__init__()
        self.is_deconv = False
        is_deconv = False
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.layer0.conv1,
                                   self.encoder.layer0.bn1,
                                   self.encoder.layer0.relu1,
                                   )
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        bottom_channel_nr = 2048
        self.center = nn.Sequential(
            Conv3BN(2048, 512, bn=True),
            Conv3BN(512, 256, bn=True),
            self.pool
        )

        self.dec5 = DecoderBlockV3(bottom_channel_nr + 256, 512, 64)
        self.dec4 = DecoderBlockV3(bottom_channel_nr // 2 + 64, 256, 64)
        self.dec3 = DecoderBlockV3(bottom_channel_nr // 4 + 64, 128, 64)
        self.dec2 = DecoderBlockV3(bottom_channel_nr // 8 + 64, 64, 64)
        self.dec1 = DecoderBlockV3(64, 32, 64)

        self.final = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        #print("x: ", x.size())
        conv1 = self.conv1(x)#;                print("conv1: ", conv1.size())
        conv2 = self.conv2(conv1)#;              print("conv2: ", conv2.size())
        conv3 = self.conv3(conv2)#;              print("conv3: ", conv3.size())
        conv4 = self.conv4(conv3)#;              print("conv4: ", conv4.size())
        conv5 = self.conv5(conv4)#;              print("conv5: ", conv5.size())

        center = self.center(conv5)#; print("center: ", center.size())
        dec5 = self.dec5(center, conv5)#;print("dec5: ", dec5.size())
        dec4 = self.dec4(dec5, conv4)#;print("dec4: ", dec4.size())
        dec3 = self.dec3(dec4, conv3)#;print("dec3: ", dec3.size())
        dec2 = self.dec2(dec3, conv2)#;print("dec2: ", dec2.size())
        dec1 = self.dec1(dec2)#;print("dec1: ", dec1.size())

        # hypercolumn
        f = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(dec3, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(dec4, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(dec5, scale_factor=16, mode="bilinear", align_corners=False),
        ), 1)
        f = F.dropout2d(f, p=0.4, training=self.training)
        x_out = self.final(f)

        return x_out




class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x



if __name__ == '__main__':
    model = SE_ResNext50_2(num_classes=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    images = torch.randn(1, 3, 128, 128).to(device)
    f = model.forward(images)

def run_check_net():

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)


    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()


    #---
    net = SaltNet().cuda()
    net.set_mode('train')

    logit = net(input)
    loss  = net.criterion(logit, truth)
    dice  = net.metric(logit, truth)

    print('loss : %0.8f'%loss.item())
    print('dice : %0.8f'%dice.item())
    print('')


    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    while i<=500:

        logit = net(input)
        loss  = net.criterion(logit, truth)
        dice  = net.metric(logit, truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] loss, dice  :  %0.5f,%0.5f'%(i, loss.item(),dice.item()))
        i = i+1