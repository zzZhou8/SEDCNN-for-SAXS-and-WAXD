import torch
import torch.nn as nn

class DecoderBlock(nn.Module):# it is DecoderBlock

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        inp = x
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = x + inp
        x = self.relu(x)

        return x

class EncoderBlock(nn.Module):# it is EncoderBlock

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size=3, padding=1,stride=1)
        self.conv1 = nn.Conv2d(int(out_channels/4), out_channels, kernel_size=3, padding=1,stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        inp = x
        x = self.conv0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = x + inp
        x = self.relu(x)

        return x

class DownsampleBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels_final: int):
        super().__init__()


        mid_channels = int(out_channels_final/4)
        out_channels = int(out_channels_final/2)
        #5*5的降采样
        self.conv5_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=2, padding=2)   #N   -- N/2
        self.conv5_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=5, stride=1, padding=2)  #N/2 -- N/2
        #3*3的降采样
        self.conv3_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)   #N   -- N/2
        self.conv3_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)  #N/2 -- N/2
        #RELU激活
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        #依赖于3*3卷积的N-- N/2的操作
        conv3 = self.relu(self.conv3_1(x))
        conv3 = self.conv3_2(conv3)

        #依赖于5*5卷积的N-- N/2的操作
        conv5 = self.relu(self.conv5_1(x))
        conv5 = self.conv5_2(conv5)
        #通道叠加
        output = torch.cat((conv3,conv5),dim=1)

        return output

def EncoderStage(in_channels: int, out_channels: int, num_blocks: int):

    #Downsample_block
    blocks = [DownsampleBlock(in_channels=in_channels,out_channels_final=out_channels)]
    #Encoder_block
    for _ in range(num_blocks-1):
        blocks.append(EncoderBlock(in_channels=out_channels,out_channels=out_channels))
    return nn.Sequential(*blocks)


class UpsampleBlock(nn.Module):

    def __init__(self, in_channels: int, skip_in_channels: int, out_channels: int):
        super().__init__()

        self.residual_conv = DecoderBlock(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, int(out_channels/2), kernel_size=2, stride=2, padding=0)#N - 2N
        self.proj_conv = nn.Conv2d(skip_in_channels, int(out_channels/2), kernel_size=3, stride=1, padding=1)#N - N
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        inp, skip = inputs
        #进行一次卷积操作
        layer_out1 = self.residual_conv(inp)                #N -- N
        #向上采样
        layer_out1 = self.upsample(layer_out1)              #N -- 2N
        #投影叠加
        layer_out2 = self.proj_conv(skip)                   #N -- N
        #通道叠加
        layer_out = torch.cat((layer_out1,layer_out2),dim=1)

        return layer_out
   
#4个核心层
class SEDCNN_WAXD(nn.Module):
    def __init__(self):
        super(SEDCNN_WAXD, self).__init__()

        self.relu = nn.ReLU()
        
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.enc1 = EncoderStage(in_channels=16, out_channels=32, num_blocks=2)
        self.enc2 = EncoderStage(in_channels=32, out_channels=64, num_blocks=2)
        self.enc3 = EncoderStage(in_channels=64, out_channels=128, num_blocks=4)
        self.enc4 = EncoderStage(in_channels=128, out_channels=256, num_blocks=4)
        
        self.encdec = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, stride=1)

        self.dec1 = UpsampleBlock(in_channels=64, skip_in_channels=128, out_channels=64)
        self.dec2 = UpsampleBlock(in_channels=64, skip_in_channels=64, out_channels=32)
        self.dec3 = UpsampleBlock(in_channels=32, skip_in_channels=32, out_channels=32)
        self.dec4 = UpsampleBlock(in_channels=32, skip_in_channels=16, out_channels=16)
        

        self.out0 = DecoderBlock(in_channels=16, out_channels=16)
        self.out1 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)
    

    def forward(self, inp):
        # Encoder

        conv0 = self.relu(self.conv0(inp))       # N    * N    * 1   --- N    * N    * 16
        conv1 = self.enc1(conv0)                 # N    * N    * 16  --- N/2  * N/2  * 32
        conv2 = self.enc2(conv1)                 # N/2  * N/2  * 32  --- N/4  * N/4  * 64
        conv3 = self.enc3(conv2)                 # N/4  * N/4  * 64  --- N/8  * N/8  * 128
        conv4 = self.enc4(conv3)                 # N/8  * N/8  * 128 --- N/16 * N/16 * 256

        conv5 = self.relu(self.encdec(conv4))    # N/16 * N/16 * 256 --- N/16 * N/16 * 64

        up3 = self.dec1((conv5, conv3))          # N/16 * N/16 * 64  --- N/8  * N/8  * 64
        up2 = self.dec2((up3, conv2))            # N/8  * N/8  * 64  --- N/4  * N/4  * 32
        up1 = self.dec3((up2, conv1))            # N/4  * N/4  * 32  --- N/2  * N/2  * 32
        x = self.dec4((up1, conv0))              # N/2  * N/2  * 32  --- N  *   N    * 16

        x = self.out0(x)
        x = self.relu(self.out1(x))

        return x
    
    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)  