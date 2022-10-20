import torch
import torch.nn as nn
from collections import OrderedDict

def Conv2D(in_channels:int,out_channels:int,kernel_size:int,stride:int,padding:int,is_seperable:bool=False,has_relu:bool=False):
    modules = OrderedDict()
    
  
    if is_seperable:
        modules['depthwise'] = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,groups=in_channels, bias=False)
        modules['pointwise'] = nn.Conv2d(in_channels,out_channels,kernel_size=1, stride=1, padding=0, bias=True)

    else:
        modules['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,bias=True)
    
    if has_relu:
        modules['relu'] = nn.ReLU()

    return nn.Sequential(modules)

class DecoderLast(nn.Module):

    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()

        self.conv0 = Conv2D(in_channels, middle_channels*2, kernel_size=3, padding=1,stride=1, is_seperable=True, has_relu=True)#96-48
        self.conv1 = Conv2D(middle_channels*2, middle_channels, kernel_size=3, padding=1,stride=1, is_seperable=True, has_relu=True)#48-24
        self.conv2 = Conv2D(middle_channels, out_channels, kernel_size=3, padding=1,stride=1, is_seperable=False, has_relu=False)#24-1

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class SEDCNN_SAXS(nn.Module):
    def __init__(self,in_channels=1,out_channels=48):
        super(SEDCNN_SAXS, self).__init__()
        self.conv_first = Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1,is_seperable=False,has_relu=True)
        self.conv = Conv2D(out_channels, out_channels, kernel_size=5, stride=1, padding=0,is_seperable=True,has_relu=True)
        self.same_conv = Conv2D(out_channels*2, out_channels*2, kernel_size=3, stride=1, padding=1,is_seperable=True,has_relu=True)
        self.conv_lesschannal=  Conv2D(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1,is_seperable=True,has_relu=True)

        self.deconv_first = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.deconv = nn.ConvTranspose2d(out_channels*2, out_channels*2, kernel_size=5, stride=1, padding=0)
        self.Deconderlast= DecoderLast(in_channels=96,middle_channels=24,out_channels=1)
        self.relu = nn.ReLU()
        self.Leakyrelu = nn.LeakyReLU(0.1)
        
    
    def forward(self, x):
        # Encoder
        inp = x
        layer = self.relu(self.conv_first(x))           #N   * N   * 1  -- N * N * 48

        residual1 = layer.clone()                       #N   * N   *48
        layer = self.relu(self.conv(layer))             #N   * N   * 1  -- N-4 * N-4 * 48
        layer = self.relu(self.conv(layer))             #N-4 * N-4 * 48 -- N-8 * N-8 * 48 

        residual2 = layer.clone()                       #N-8 * N-8 * 48
        layer = self.relu(self.conv(layer))             #N-8 * N-8 * 48 -- N-12 * N-12 * 48
        layer = self.relu(self.conv(layer))             #N-12 * N-12 * 48 -- N-16 * N-16 * 48 

        residual3 = layer.clone()                       #N-16 * N-16 * 48 
        layer = self.relu(self.conv(layer))             #N-16 * N-16 * 48 -- N-20 * N-20 * 48 
        layer = self.relu(self.conv(layer))             #N-20 * N-20 * 48 -- N-24 * N-24 * 48 

        residual4 = layer.clone()                       #N-24 * N-24 * 48  
        layer = self.relu(self.conv(layer))             #N-24 * N-24 * 48 -- N-28 * N-28 * 48 
        layer = self.relu(self.conv(layer))             #N-28 * N-28 * 48 -- N-32 * N-32 * 48 

        residual5 = layer.clone()                       #N-32 * N-32 * 48
        layer = self.relu(self.conv(layer))             #N-32 * N-32 * 48 -- N-36 * N-36 * 48 

        # decoder
        
        layer = self.relu(self.deconv_first(layer))     #N-36 * N-36 * 48 -- N-32 * N-32 * 48

        layer = torch.cat((layer,residual5),dim=1)      #N-32 * N-32 * 96
        layer = self.relu(self.same_conv(layer))        #N-32 * N-32 * 96 -- N-32 * N-32 * 96
        layer = self.relu(self.deconv(layer))           #N-32 * N-32 * 96 -- N-28 * N-28 * 96
        layer = self.relu(self.deconv(layer))           #N-28 * N-28 * 96 -- N-24 * N-24 * 96
        layer = self.relu(self.conv_lesschannal(layer)) #N-24 * N-24 * 96 -- N-24 * N-24 * 48

        layer = torch.cat((layer,residual4),dim=1)      #N-24 * N-24 * 96
        layer = self.relu(self.same_conv(layer))        #N-24 * N-24 * 96 -- N-24 * N-24 * 96
        layer = self.relu(self.deconv(layer))           #N-24 * N-24 * 96 -- N-20 * N-20 * 96
        layer = self.relu(self.deconv(layer))           #N-20 * N-20 * 96 -- N-16 * N-16 * 96
        layer = self.relu(self.conv_lesschannal(layer)) #N-16 * N-16 * 96 -- N-16 * N-16 * 48

        layer = torch.cat((layer,residual3),dim=1)      #N-16 * N-16 * 96
        layer = self.relu(self.same_conv(layer))        #N-16 * N-16 * 96 -- N-16 * N-16 * 96
        layer = self.relu(self.deconv(layer))           #N-16 * N-16 * 96 -- N-12 * N-12 * 96
        layer = self.relu(self.deconv(layer))           #N-12 * N-12 * 96   -- N-8 * N-8 * 96
        layer = self.relu(self.conv_lesschannal(layer)) #N-8 * N-8 * 96   -- N-8 * N-8 * 48

        layer = torch.cat((layer,residual2),dim=1)      #N-8 * N-8 * 48*2
        layer = self.relu(self.same_conv(layer))        #N-8 * N-8 * 96 -- N-8 * N-8 * 96
        layer = self.relu(self.deconv(layer))           #N-8 * N-8 * 96 -- N-4 * N-4 * 96
        layer = self.relu(self.deconv(layer))           #N-4 * N-4 * 96 -- N * N * 96
        layer = self.relu(self.conv_lesschannal(layer)) #N   * N   * 96 -- N * N * 48
        
        layer = torch.cat((layer,residual1),dim=1)      #N * N * 96
        layer = self.relu(self.same_conv(layer))        #N * N * 96 -- N * N * 96
        layer = self.Deconderlast(layer)                #N * N * 96 -- N * N * 1
        #layer = self.Leakyrelu(layer)                   #N * N * 1 -- N * N * 1
        pred = layer+inp
        return pred
    
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
