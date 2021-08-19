import torch

#batch normalization is set to False
#5 residual layers

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
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



class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        
        #TODO: 3x3 convolution -> relu
        #the input and output channel number is channel_num
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect'),
            torch.nn.InstanceNorm2d(out_channels),
            torch.nn.ReLU(),
        ) 
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False, padding_mode='reflect'),
            torch.nn.InstanceNorm2d(out_channels),
        )
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)
    
    def forward(self, x):
        
        #TODO: forward
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return self.drop(out)


class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7,stride=1, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2, kernel_size=3, stride=2,batch_norm=False)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, kernel_size=3, stride=2,batch_norm=False)

        self.residual1 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual2 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual3 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual4 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual5 = BasicBlock(num_filter * 4, num_filter * 4)
        # self.residual2 = ResBlock(num_filter * 4, num_filter * 4)
        # self.residual3 = ResBlock(num_filter * 4, num_filter * 4)

        self.deconv1 = DeconvBlock(num_filter * 4, num_filter * 2,kernel_size=3,stride=2,batch_norm=False)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter, kernel_size=3,stride=2,batch_norm=False)
        self.deconv3 = DeconvBlock(num_filter, output_dim, kernel_size=10, stride=1, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)


        res1 = self.residual1(enc3)
        res2 = self.residual2(res1)
        res3 = self.residual3(res2)
        res4 = self.residual4(res3)
        res5 = self.residual5(res4)
        # res2 = self.residual2(res1)
        # res3 = self.residual3(res2)

        dec1 = self.deconv1(res5)
        dec2 = self.deconv2(dec1)
        dec3 = self.deconv3(dec2)

        out = torch.nn.Tanh()(dec3)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)

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

