import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pdb

BN_MOMENTUM = 0.1

class Resampling(nn.Module):
    '''多尺度融合时候一样的特征图直接输出'''
    def __init__(self):
        super(Resampling, self).__init__()
    def forward(self, x):
        return x

class HRNetConv3x3(nn.Module):
    def __init__(self, input_chs, output_chs, kernel_size=3, stride=1, padding=0):
        super(HRNetConv3x3, self).__init__()
        self.Conv3x3 = nn.Sequential(
            nn.Conv2d(input_chs, output_chs, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_chs, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.Conv3x3(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.bottleneck(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        output = self.relu(x)
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.bottleneck(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        output = self.relu(x)
        return output

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_branches = num_branches
        self.num_inchannels = num_inchannels
        self.num_channels = num_channels
        self.fuse_method = fuse_method
        self.multi_scale_output = multi_scale_output

        self.branches = self.make_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.fuse_layers = self.make_fuse_layers()
        self.relu = nn.ReLU(False)

    def check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
                        
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            raise ValueError(error_msg)
        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            raise ValueError(error_msg)
        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def make_branches(self, num_branches, block, num_blocks, num_inchannels, num_channels):
        branches = []
        for i in range(num_branches):
            # 每个branch 是横向的四个block
            branches.append(self.make_one_branch(i, block, num_blocks, num_inchannels, num_channels))
        return nn.ModuleList(branches)

    def make_one_branch(self, branch_index, block, num_blocks, num_inchannels, num_channels, stride=1):
        downsample = None
        if stride != 1 or num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(num_inchannels[branch_index], num_channels[branch_index] * block.expansion, 
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self, cfg):
        super(HighResolutionNet, self).__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            # 通过Stem得到1/4 resoluation 的特征图
            HRNetConv3x3(3, 64, kernel_size=3, stride=2, padding=1),
            HRNetConv3x3(64, 64, kernel_size=3, stride=2, padding=1),
        )
        # Branch 1 2 3 4 分别对应原始图像 1/4，1/8， 1/16, 1/32 的分辨率
        # stage 1
        num_channels = self.cfg['STAGE1']['NUM_CHANNELS'][0]
        num_blocks = self.cfg['STAGE1']['NUM_BLOCKS'][0]
        self.layer1 = self.make_layer(Bottleneck, 64, num_channels, num_blocks)
        stage1_out_channel = Bottleneck.expansion*num_channels
        # stage 2
        num_channels = self.cfg['STAGE2']['NUM_CHANNELS']
        num_channels = [num_channels[i] * BasicBlock.expansion for i in range(len(num_channels))]
        self.transition1 = self.make_transition_layer( [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self.make_stage(self.cfg['STAGE2'], BasicBlock, num_channels)
        # stage 3
        num_channels = self.cfg['STAGE3']['NUM_CHANNELS']
        num_channels = [num_channels[i] * BasicBlock.expansion for i in range(len(num_channels))]
        self.transition2 = self.make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self.make_stage(self.cfg['STAGE3'], BasicBlock, num_channels)
        # stage 4
        num_channels = self.cfg['STAGE4']['NUM_CHANNELS']
        num_channels = [num_channels[i] * BasicBlock.expansion for i in range(len(num_channels))]
        self.transition3 = self.make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self.make_stage(self.cfg['STAGE4'], BasicBlock, num_channels)

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self.make_Cls_head(pre_stage_channels)
        self.classifier = nn.Linear(2048, 1000)
        # # HR Representation Head
        # self.HRNetV2 = self.make_HRNetV2_head()
        self.init_weights()

    def forward(self, x):
        # input the image into a stem -> [64, 56, 56]
        x = self.stem(x)
        # stage 1 -> [256, 56, 56]
        x = self.layer1(x)
        # stage 2 -> [64, 56, 56][128, 28, 28]
        x_list = []
        for i in range(2):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        # stage 3 -> [64, 56, 56][128, 28, 28][256, 14, 14]
        x_list = []
        for i in range(3):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        # stage 4 -> [64, 56, 56][128, 28, 28][256, 14, 14][512, 7, 7]
        x_list = []
        for i in range(4):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            y = F.avg_pool2d(y, kernel_size=y.size()
                                 [2:]).view(y.size(0), -1)
        y = self.classifier(y)

        # # HR Representation
        # y0 = HRNetV2[0](y_list[0])
        # y1 = HRNetV2[1](y_list[1])
        # y2 = HRNetV2[2](y_list[2])
        # y3 = HRNetV2[3](y_list[3])
        # y = torch.cat((y0, y1, y2, y3), 1)

        return y

    def make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    
    def make_transition_layer(self, pre_stage_channels, cur_stage_channels):
        # pre_stage_channels is [branch1, branch2]
        # cur_stage_channels is [branch1, branch2, branch3]
        num_branches_pre = len(pre_stage_channels)
        num_branches_cur = len(cur_stage_channels)

        # 生成分支拓展
        transition_layers = []
        for i in range(num_branches_cur):
 
            if i < num_branches_pre:
                if cur_stage_channels[i] != pre_stage_channels[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(pre_stage_channels[i], cur_stage_channels[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cur_stage_channels[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    #transition_layers.append(Resampling()) #也可以
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = pre_stage_channels[-1]
                    outchannels = cur_stage_channels[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def make_stage(self, layer_cfg, BLOCK, num_inchannels, multi_scale_output=True):
        num_modules = layer_cfg['NUM_MODULES']
        num_branches = layer_cfg['NUM_BRANCHES']
        num_blocks = layer_cfg['NUM_BLOCKS']
        num_channels = layer_cfg['NUM_CHANNELS']
        fuse_method = layer_cfg['FUSE_METHOD']
        block = BLOCK

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def make_HRNetV2_head(self):
        '''聚合输出 High Resolution Representations'''
        layers = []
        for i in range(4):
            sampling = nn.Upsample(scale_factor=2**(i), mode='nearest')
            layers.append(sampling)
        return nn.Sequential(*layers)  

    def make_Cls_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self.make_layer(head_block, channels, head_channels[i], 1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(in_channels=head_channels[3] * head_block.expansion, out_channels=2048, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained_model(self, dir):
        pretrained_dict = torch.load(dir)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
            

def HRNet_W18_C_Small_v2():
    '''
    #Params 15.6M
    GFLOPs 2.42	
    top-1 error 24.9%
    top-5 error 7.6%
    '''
    cfg = {
            'STAGE1':{'NUM_MODULES':1, 'NUM_BRANCHES': 1, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[2], 'NUM_CHANNELS':[18], 'FUSE_METHOD': 'SUM'},
            'STAGE2':{'NUM_MODULES':1, 'NUM_BRANCHES': 2, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[2,2], 'NUM_CHANNELS':[18,36], 'FUSE_METHOD': 'SUM'},
            'STAGE3':{'NUM_MODULES':3, 'NUM_BRANCHES': 3, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[2,2,2], 'NUM_CHANNELS':[18,36,72], 'FUSE_METHOD': 'SUM'},
            'STAGE4':{'NUM_MODULES':2, 'NUM_BRANCHES': 4, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[2,2,2,2], 'NUM_CHANNELS':[18,36,72,144], 'FUSE_METHOD': 'SUM'}
            }
    model = HighResolutionNet(cfg)
    return model

def HRNet_W30_C():
    '''
    #Params 37.7M
    GFLOPs 7.55
    top-1 error 21.8%
    top-5 error 5.8%
    '''
    cfg = {
            'STAGE1':{'NUM_MODULES':1, 'NUM_BRANCHES': 1, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_CHANNELS':[30], 'FUSE_METHOD': 'SUM'},
            'STAGE2':{'NUM_MODULES':1, 'NUM_BRANCHES': 2, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4], 'NUM_CHANNELS':[30,60], 'FUSE_METHOD': 'SUM'},
            'STAGE3':{'NUM_MODULES':4, 'NUM_BRANCHES': 3, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4], 'NUM_CHANNELS':[30,60,120], 'FUSE_METHOD': 'SUM'},
            'STAGE4':{'NUM_MODULES':3, 'NUM_BRANCHES': 4, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4,4], 'NUM_CHANNELS':[30,60,120,240], 'FUSE_METHOD': 'SUM'}
            }
    model = HighResolutionNet(cfg)
    if pretrained:
        model.load_pretrained_model(modelPath)
    return model

def HRNet_W40_C():
    '''
    #Params 57.6M
    GFLOPs 11.8
    top-1 error 21.1%
    top-5 error 5.5%
    '''
    cfg = {
            'STAGE1':{'NUM_MODULES':1, 'NUM_BRANCHES': 1, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_CHANNELS':[40], 'FUSE_METHOD': 'SUM'},
            'STAGE2':{'NUM_MODULES':1, 'NUM_BRANCHES': 2, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4], 'NUM_CHANNELS':[40,80], 'FUSE_METHOD': 'SUM'},
            'STAGE3':{'NUM_MODULES':4, 'NUM_BRANCHES': 3, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4], 'NUM_CHANNELS':[40,80,160], 'FUSE_METHOD': 'SUM'},
            'STAGE4':{'NUM_MODULES':3, 'NUM_BRANCHES': 4, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4,4], 'NUM_CHANNELS':[40,80,160,320], 'FUSE_METHOD': 'SUM'}
            }
    model = HighResolutionNet(cfg)
    return model

def HRNet_W48_C():
    '''
    #Params 57.6M
    GFLOPs 11.8
    top-1 error 21.1%
    top-5 error 5.5%
    '''
    cfg = {
            'STAGE1':{'NUM_MODULES':1, 'NUM_BRANCHES': 1, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_CHANNELS':[48], 'FUSE_METHOD': 'SUM'},
            'STAGE2':{'NUM_MODULES':1, 'NUM_BRANCHES': 2, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4], 'NUM_CHANNELS':[48,96], 'FUSE_METHOD': 'SUM'},
            'STAGE3':{'NUM_MODULES':4, 'NUM_BRANCHES': 3, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4], 'NUM_CHANNELS':[48,96,192], 'FUSE_METHOD': 'SUM'},
            'STAGE4':{'NUM_MODULES':3, 'NUM_BRANCHES': 4, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4,4], 'NUM_CHANNELS':[48,96,192,384], 'FUSE_METHOD': 'SUM'}
            }
    model = HighResolutionNet(cfg)
    return model

def HRNet_W64_C(pretrained=False):
    '''
    #Params 57.6M
    GFLOPs 11.8
    top-1 error 21.1%
    top-5 error 5.5%
    '''
    curdir = os.getcwd()
    modelPath = os.path.join(curdir, 'HRnetPretrainModel', 'hrnetv2_w64_imagenet_pretrained.pth')
    cfg = {
            'STAGE1':{'NUM_MODULES':1, 'NUM_BRANCHES': 1, 'BLOCK':'BOTTLENECK', 'NUM_BLOCKS':[4], 'NUM_CHANNELS':[64], 'FUSE_METHOD': 'SUM'},
            'STAGE2':{'NUM_MODULES':1, 'NUM_BRANCHES': 2, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4], 'NUM_CHANNELS':[64,128], 'FUSE_METHOD': 'SUM'},
            'STAGE3':{'NUM_MODULES':4, 'NUM_BRANCHES': 3, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4], 'NUM_CHANNELS':[64,128,256], 'FUSE_METHOD': 'SUM'},
            'STAGE4':{'NUM_MODULES':3, 'NUM_BRANCHES': 4, 'BLOCK':'BasicBlock', 'NUM_BLOCKS':[4,4,4,4], 'NUM_CHANNELS':[64,128,256,512], 'FUSE_METHOD': 'SUM'}
            }
    model = HighResolutionNet(cfg)
    if pretrained:
        model.load_pretrained_model(modelPath)
    return model

if __name__ == "__main__":
    pass
    model = HRNet_W64_C(pretrained=True)
    x = torch.rand(size=(1,3,224,224), dtype=torch.float32)
    x = model(x)
    print(x.shape)  
    #pdb.set_trace()