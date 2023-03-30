import torch
from torch import nn
from torch.utils import model_zoo
import torch.nn.functional as F
from batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 512
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(256*5, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ScalePyramidModule1(nn.Module):
    def __init__(self):
        super(ScalePyramidModule1, self).__init__()
        self.assp = ASPP(512, output_stride=16, BatchNorm=SynchronizedBatchNorm2d)
        self.assp1 = ASPP(512, output_stride=8, BatchNorm=SynchronizedBatchNorm2d)


    def forward(self, *input1):
        #conv2_2, conv3_3, conv4_3, conv5_3 = input
        conv2_2, conv3_3, conv4_3, conv5_3 = input1


        conv5_31 = self.assp(conv5_3)
        conv4_31 = self.assp1(conv4_3)
        conv2_21 = conv2_2
        conv3_31 = conv3_3

        return conv2_21, conv3_31, conv4_31, conv5_31


class ScalePyramidModule(nn.Module):
    def __init__(self):
        super(ScalePyramidModule, self).__init__()
        #self.assp = ASPP(512, output_stride=16, BatchNorm=SynchronizedBatchNorm2d)
        #self.can = ContextualModule(512, 512)

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        #conv4_3 = self.can(conv4_3)
        #conv5_3 = self.assp(conv5_3)

        return conv2_2, conv3_3, conv4_3, conv5_3

class Model(nn.Module):#总定义
    def __init__(self):
        super(Model, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.amp = BackEnd1()
        self.dmp = BackEnd()
        self.spm1 = ScalePyramidModule1()
        self.spm = ScalePyramidModule()

        self.conv_att = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=True)
        self.conv_out = BaseConv(32, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):#总结构顺序
        input = self.vgg(input)
        #input1 = self.vgg(input1)
        spm_out = self.spm(*input)
        spm_out1 = self.spm1(*input)
        amp_out = self.amp(*spm_out1)
        dmp_out = self.dmp(*spm_out)

        amp_out = self.conv_att(amp_out)
        dmp_out = amp_out * dmp_out
        dmp_out = self.conv_out(dmp_out)

        return dmp_out, amp_out

    def load_vgg(self):#VGG加载
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)


class VGG(nn.Module):#VGG定义
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):#VGG结构顺序
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)#(2/H,2/W,128)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)#(4/H,4/W,256)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)#(8/H,8/W,512)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)#(16/H.16/W,512)

        return conv2_2, conv3_3, conv4_3, conv5_3


class SAModule(nn.Module):#SA模块定义
    def __init__(self, in_channels, out_channels, use_bn=True):
        super(SAModule, self).__init__()
        branch_out = out_channels // 4
        self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
                                   kernel_size=1)
        self.branch3x3 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=3, padding=1),
        )
        self.branch5x5 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=5, padding=2),
        )
        self.branch7x7 = nn.Sequential(
            BasicConv(in_channels, 2 * branch_out, use_bn=use_bn,
                      kernel_size=1),
            BasicConv(2 * branch_out, branch_out, use_bn=use_bn,
                      kernel_size=7, padding=3),
        )

    def forward(self, x):#SA模块结构
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch7x7 = self.branch7x7(x)
        out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
        return out


class BackEnd1(nn.Module):
    def __init__(self):
        super(BackEnd1, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv1 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3 = BaseConv(768, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input1):
        conv2_21, conv3_31, conv4_31, conv5_31 = input1

        input1 = self.upsample(conv5_31)# （H/8,256）

        input1 = torch.cat([input1, conv4_31], 1)#(H/8,512)
        input1 = self.conv1(input1)#(H/8,256)
        input1 = self.conv2(input1)#(H/8,256)
        input1 = self.upsample(input1)#(H/4,256)

        input1 = torch.cat([input1, conv3_31, self.upsample4(conv5_31)], 1)#(H/4,768)
        input1 = self.conv3(input1)#(H/4,128)
        input1= self.conv4(input1)#(H/4,128)
        input1 = self.upsample(input1)#(H/2,128)

        input1 = torch.cat([input1, conv2_21], 1)#(H/2,256)
        input1= self.conv5(input1)#(H/2,64)
        input1 = self.conv6(input1)#(H/2,64)
        input1 = self.conv7(input1)#(H/2,32)

        return input1
class BackEnd(nn.Module):#密度图后端定义
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.SAModule3 = SAModule(1024,256, use_bn=True)
        self.SAModule2 = SAModule(512, 128, use_bn=True)
        self.SAModule1 = SAModule(256, 64, use_bn=True)
        #self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        #self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        #self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        #self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        #self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        #self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, *input):#密度图后端结构
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = self.upsample(conv5_3)
        input = torch.cat([input, conv4_3], 1)
        input = self.SAModule3(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_3], 1)
        input = self.SAModule2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.SAModule1(input)
        input = self.conv7(input)
        
        return input
        #input = torch.cat([input, conv4_3], 1)
        #input = self.conv1(input)
        #input = self.conv2(input)
        #input = self.upsample(input)

        #input = torch.cat([input, conv3_3], 1)
        #input = self.conv3(input)
        #input = self.conv4(input)
        #input = self.upsample(input)

        #input = torch.cat([input, conv2_2], 1)
        #input = self.conv5(input)
        #input = self.conv6(input)
        #input = self.conv7(input)

        #return input

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
        super(BasicConv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return F.relu(x, inplace=True)


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__ == '__main__':
    input = torch.randn(8, 3, 400, 400).cuda()
    model = Model().cuda()
    output, attention = model(input)
    print(input.size())
    print(output.size())
    print(attention.size())