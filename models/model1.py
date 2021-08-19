import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=0, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ResBlock, self).__init__()

        self.skip = torch.nn.Sequential()

        if stride != 1 or in_channels != out_channels:
          self.skip = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels))
        else:
          self.skip = None

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect'),
            torch.nn.InstanceNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect'),
            torch.nn.InstanceNorm2d(out_channels))

        self.relu = torch.nn.ReLU(True)


    def forward(self, x):
        identity = x
        out = self.block(x)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        out = self.relu(out)

        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out


# class Generator(torch.nn.Module):
#     def __init__(self, input_dim, num_filter, output_dim):
#         super(Generator, self).__init__()

#         # Encoder
#         self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
#         self.conv2 = ConvBlock(num_filter, num_filter * 2)
#         self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
#         self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
#         self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
#         self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
#         self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
#         self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        

#         self.residual1 = ResBlock(num_filter * 8, num_filter * 8)
#         self.residual2 = ResBlock(num_filter * 8, num_filter * 8)
#         self.residual3 = ResBlock(num_filter * 8, num_filter * 8)

#         # Decoder
#         self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
#         self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
#         self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
#         self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
#         self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
#         self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
#         self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)
#         self.deconv8 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.conv1(x)
#         enc2 = self.conv2(enc1)
#         enc3 = self.conv3(enc2)
#         enc4 = self.conv4(enc3)
#         enc5 = self.conv5(enc4)
#         enc6 = self.conv6(enc5)
#         enc7 = self.conv7(enc6)
#         enc8 = self.conv8(enc7)

#         res1 = self.residual1(enc8)
#         res2 = self.residual2(res1)
#         res3 = self.residual3(res2)

#         # Decoder with skip-connections
#         dec1 = self.deconv1(res3)
#         dec1 = torch.cat([dec1, enc7], 1)
#         dec2 = self.deconv2(dec1)
#         dec2 = torch.cat([dec2, enc6], 1)
#         dec3 = self.deconv3(dec2)
#         dec3 = torch.cat([dec3, enc5], 1)
#         dec4 = self.deconv4(dec3)
#         dec4 = torch.cat([dec4, enc4], 1)
#         dec5 = self.deconv5(dec4)
#         dec5 = torch.cat([dec5, enc3], 1)
#         dec6 = self.deconv6(dec5)
#         dec6 = torch.cat([dec6, enc2], 1)
#         dec7 = self.deconv7(dec6)
#         dec7 = torch.cat([dec7, enc1], 1)
#         dec8 = self.deconv8(dec7)
#         out = torch.nn.Tanh()(dec8)
#         return out

#     def normal_weight_init(self, mean=0.0, std=0.02):
#         for m in self.children():
#             if isinstance(m, ConvBlock):
#                 torch.nn.init.normal(m.conv.weight, mean, std)
#             if isinstance(m, DeconvBlock):
#                 torch.nn.init.normal(m.deconv.weight, mean, std)


class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7,stride=1, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2, kernel_size=3, stride=2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, kernel_size=3, stride=2)

        # self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        # self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        # self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        # self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        # self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        

        self.residual1 = ResBlock(num_filter * 4, num_filter * 4)
        self.residual2 = ResBlock(num_filter * 4, num_filter * 4)
        self.residual3 = ResBlock(num_filter * 4, num_filter * 4)

        self.deconv1 = DeconvBlock(num_filter * 4, num_filter * 2,kernel_size=3,stride=2)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter, kernel_size=3,stride=2)
        self.deconv3 = DeconvBlock(num_filter, output_dim, kernel_size=10, stride=1, batch_norm=False)
        

        # Decoder
        # self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        # self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        # self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        # self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        # self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        # self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        # self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)
        # self.deconv8 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)


        res1 = self.residual1(enc3)
        res2 = self.residual2(res1)
        res3 = self.residual3(res2)

        dec1 = self.deconv1(res3)
        dec2 = self.deconv2(dec1)
        dec3 = self.deconv3(dec2)

        out = torch.nn.Tanh()(dec3)
        return out

        # enc4 = self.conv4(enc3)
        # enc5 = self.conv5(enc4)
        # enc6 = self.conv6(enc5)
        # enc7 = self.conv7(enc6)
        # enc8 = self.conv8(enc7)

        # res1 = self.residual1(enc8)
        # res2 = self.residual2(res1)
        # res3 = self.residual3(res2)

        # # Decoder with skip-connections
        # dec1 = self.deconv1(res3)
        # dec1 = torch.cat([dec1, enc7], 1)
        # dec2 = self.deconv2(dec1)
        # dec2 = torch.cat([dec2, enc6], 1)
        # dec3 = self.deconv3(dec2)
        # dec3 = torch.cat([dec3, enc5], 1)
        # dec4 = self.deconv4(dec3)
        # dec4 = torch.cat([dec4, enc4], 1)
        # dec5 = self.deconv5(dec4)
        # dec5 = torch.cat([dec5, enc3], 1)
        # dec6 = self.deconv6(dec5)
        # dec6 = torch.cat([dec6, enc2], 1)
        # dec7 = self.deconv7(dec6)
        # dec7 = torch.cat([dec7, enc1], 1)
        # dec8 = self.deconv8(dec7)
        # out = torch.nn.Tanh()(dec8)
        # return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)


# class Generator128(torch.nn.Module):
#     def __init__(self, input_dim, num_filter, output_dim):
#         super(Generator128, self).__init__()

#         # Encoder
#         self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
#         self.conv2 = ConvBlock(num_filter, num_filter * 2)
#         self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
#         self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
#         self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
#         self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
#         self.conv7 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
#         # Decoder
#         self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
#         self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
#         self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
#         self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
#         self.deconv5 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
#         self.deconv6 = DeconvBlock(num_filter * 2 * 2, num_filter)
#         self.deconv7 = DeconvBlock(num_filter * 2, output_dim, batch_norm=False)

#     def forward(self, x):
#         # Encoder
#         enc1 = self.conv1(x)
#         enc2 = self.conv2(enc1)
#         enc3 = self.conv3(enc2)
#         enc4 = self.conv4(enc3)
#         enc5 = self.conv5(enc4)
#         enc6 = self.conv6(enc5)
#         enc7 = self.conv7(enc6)
#         # Decoder with skip-connections
#         dec1 = self.deconv1(enc7)
#         dec1 = torch.cat([dec1, enc6], 1)
#         dec2 = self.deconv2(dec1)
#         dec2 = torch.cat([dec2, enc5], 1)
#         dec3 = self.deconv3(dec2)
#         dec3 = torch.cat([dec3, enc4], 1)
#         dec4 = self.deconv4(dec3)
#         dec4 = torch.cat([dec4, enc3], 1)
#         dec5 = self.deconv5(dec4)
#         dec5 = torch.cat([dec5, enc2], 1)
#         dec6 = self.deconv6(dec5)
#         dec6 = torch.cat([dec6, enc1], 1)
#         dec7 = self.deconv7(dec6)
#         out = torch.nn.Tanh()(dec7)
#         return out

#     def normal_weight_init(self, mean=0.0, std=0.02):
#         for m in self.children():
#             if isinstance(m, ConvBlock):
#                 torch.nn.init.normal(m.conv.weight, mean, std)
#             if isinstance(m, DeconvBlock):
#                 torch.nn.init.normal(m.deconv.weight, mean, std)


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)


class Discriminator128(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator128, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, stride=1)
        self.conv4 = ConvBlock(num_filter * 4, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)



