import torch

#batch normalization is set to False & instance normalization is true
#8 residual layers
#4 convolution/deconvolution layers has the u net structure

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True,inst_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.inst_norm = inst_norm
        self.instN = torch.nn.InstanceNorm2d(output_size)

    def forward(self, x):
        out = self.conv(x)

        if self.inst_norm:
            out = self.instN(out)
        # elif self.batch_norm:
        #     out = self.bn(out)

        if self.activation:
            out = self.lrelu(out)

        return out

        


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, inst_norm=True,activation=True,dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.inst_norm = inst_norm
        self.instN = torch.nn.InstanceNorm2d(output_size)

    def forward(self, x):
        out = self.deconv(x)

        if self.inst_norm:
            out = self.instN(out)        
        # elif self.batch_norm:
        #     out = self.bn(out)

        if self.activation:
            out = self.relu(out)

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
        #self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.3)
    
    def forward(self, x):
        
        #TODO: forward
        residual = x
        x = self.conv_block1(x)
        x = self.drop(x)
        x = self.conv_block2(x)
        x = x + residual
        # out = self.relu(x)
        return x


class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7,stride=1,padding=3, batch_norm=False,inst_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter*2, kernel_size=5,stride=1,padding=2, batch_norm=False,inst_norm=False)
        self.conv3 = ConvBlock(num_filter*2, num_filter * 2,batch_norm=False,inst_norm=False)
        self.conv4 = ConvBlock(num_filter * 2, num_filter * 4,batch_norm=False,inst_norm=False)

        self.residual1 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual2 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual3 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual4 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual5 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual6 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual7 = BasicBlock(num_filter * 4, num_filter * 4)
        self.residual8 = BasicBlock(num_filter * 4, num_filter * 4)


        self.deconv1 = DeconvBlock(num_filter * 4 * 2, num_filter * 2,batch_norm=False,inst_norm=False)
        self.deconv2 = DeconvBlock(num_filter * 2 * 2, num_filter * 2, batch_norm=False,inst_norm=False)
        self.deconv3 = DeconvBlock(num_filter * 2 * 2, num_filter, kernel_size=5,stride=1,padding=2, batch_norm=False,inst_norm=False)
        self.deconv4 = DeconvBlock(num_filter*2, output_dim, kernel_size=7, stride=1,padding=3,activation=False, batch_norm=False,inst_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)

        res1 = self.residual1(enc4)
        res2 = self.residual2(res1)
        res3 = self.residual3(res2)
        res4 = self.residual4(res3)
        res5 = self.residual5(res4)
        res6 = self.residual6(res5)
        res7 = self.residual7(res6)
        res8 = self.residual8(res7)

        res8 = torch.cat([res8, enc4], 1)

        dec1 = self.deconv1(res8)
        dec1 = torch.cat([dec1, enc3], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc2], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc1], 1)
        dec4 = self.deconv4(dec3)

        out = torch.nn.Tanh()(dec4)
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

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False,inst_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2, batch_norm=False)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4, batch_norm=False)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, stride=1,batch_norm=False)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False,inst_norm=False)

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

