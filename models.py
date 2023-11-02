import warnings
import torch
import torchvision
from torch import nn
from torchvision import models
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

class MLPNet(nn.Module):
    def __init__(self, class_number=10):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, class_number)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Resnet_cls(nn.Module):
    def __init__(self, orig_resnet, class_number=1000):
        super(Resnet_cls, self).__init__()
        
        self.f_dim = 2048
        self.res34and18_f_dim = 512
        self.input_dim = 224
        self.class_number = class_number
        
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.avgpool = orig_resnet.avgpool
        # orig_resnet.fc.out_features = class_number
        # self.fc = orig_resnet.fc
        self.fc = nn.Linear(orig_resnet.fc.in_features, self.class_number)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_feature_maps=False, softmax=False):
        conv_out = {}
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); conv_out['layer1'] = x.clone();
        x = self.layer2(x); conv_out['layer2'] = x.clone();
        x = self.layer3(x); conv_out['layer3'] = x.clone();
        x = self.layer4(x); conv_out['layer4'] = x.clone();
        x = self.avgpool(x)
        x = torch.flatten(x, 1); conv_out['flatten'] = x.clone();
        x = self.fc(x)
        
        if return_feature_maps:
            return conv_out
        
        if softmax:
            #normalized_masks = torch.nn.functional.softmax(x, dim=1)
            normalized_masks = self.softmax(x)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks, 'x':x}
        else:
            return x
        
        
class AlexNet_cls(nn.Module):
    def __init__(self, orig_alexnet, class_number=1000):
        super(AlexNet_cls, self).__init__()
        
        self.f_dim = 4096
        self.input_dim = 224
        self.class_number = class_number
        
        self.features = orig_alexnet.features
        self.avgpool = orig_alexnet.avgpool
        self.classifier = orig_alexnet.classifier
        self.cls_features = orig_alexnet.classifier[:-1]
        
        self.classifier[-1] = nn.Linear(self.classifier[-1].in_features, self.class_number)

    #def forward(self, x, cls_features=False, avgf=False):
    def forward(self, x, return_feature_maps=False, softmax=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1); feature_maps = x.clone()
        x = self.classifier(x)
        
        if return_feature_maps:
            return feature_maps
        
        if softmax:
            normalized_masks = torch.nn.functional.softmax(x, dim=1)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks}
        else:
            return x
        
        
class VGGs_cls(nn.Module):
    def __init__(self, orig_vgg, class_number=1000):
        super(VGGs_cls, self).__init__()
        
        self.f_dim = 512*7*7
        self.input_dim = 224
        self.class_number = class_number
        
        # take pretrained resnet, except AvgPool and FC
        self.features = orig_vgg.features
        self.avgpool = orig_vgg.avgpool
        # self.cls_features = orig_vgg.classifier[:-1]
        self.classifier = orig_vgg.classifier
        self.cls_features = orig_vgg.classifier[:-1]
        self.classifier[-1] = nn.Linear(self.classifier[-1].in_features, self.class_number)

    # def forward(self, x, cls_features=False, avgf=False):
    def forward(self, x, return_feature_maps=False, softmax=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if return_feature_maps:
            feature_maps = self.cls_features(x)
            return feature_maps
        
        x = self.classifier(x)
        if softmax:
            normalized_masks = torch.nn.functional.softmax(x, dim=1)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks}
        else:
            return x
        
        
class Squeezenets_cls(nn.Module):
    def __init__(self, orig_squeezenet, class_number=1000):
        super(Squeezenets_cls, self).__init__()
        
        self.f_dim = 512*13*13
        self.input_dim = 224
        self.class_number = class_number
        
        self.features = orig_squeezenet.features
        self.classifier = orig_squeezenet.classifier
        self.classifier[1] = nn.Conv2d(self.classifier[1].in_channels, self.class_number, kernel_size=1)

    def forward(self, x, return_feature_maps=False, softmax=False):
        x = self.features(x); feature_maps = x.clone()
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        if return_feature_maps:
            return feature_maps
        
        if softmax:
            normalized_masks = torch.nn.functional.softmax(x, dim=1)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks}
        else:
            return x
        

class Densenets_cls(nn.Module):
    def __init__(self, orig_densenet, class_number=1000):
        super(Densenets_cls, self).__init__()
        
        self.input_dim = 224
        self.class_number = class_number
        
        self.features = orig_densenet.features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = orig_densenet.classifier
        self.classifier = nn.Linear(self.classifier.in_features, self.class_number)

    #def forward(self, x, avgf=False):
    def forward(self, x, return_feature_maps=False, softmax=False):
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
            
        if return_feature_maps:
            return features
        
        if softmax:
            normalized_masks = torch.nn.functional.softmax(out, dim=1)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks}
        else:
            return out
        
        
class Inception_v3_cls(nn.Module):
    def __init__(self, orig_inception_v3, class_number=1000, aux_logits=False):
        super(Inception_v3_cls, self).__init__()
        
        self.f_dim = 2048
        self.input_dim = 229
        self.class_number = class_number
        
        # take pretrained resnet, except AvgPool and FC
        self.Conv2d_1a_3x3 = orig_inception_v3.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = orig_inception_v3.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = orig_inception_v3.Conv2d_2b_3x3
        self.maxpool1 = orig_inception_v3.maxpool1
        self.Conv2d_3b_1x1 = orig_inception_v3.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = orig_inception_v3.Conv2d_4a_3x3
        self.maxpool2 = orig_inception_v3.maxpool2
        self.Mixed_5b = orig_inception_v3.Mixed_5b
        self.Mixed_5c = orig_inception_v3.Mixed_5c
        self.Mixed_5d = orig_inception_v3.Mixed_5d
        self.Mixed_6a = orig_inception_v3.Mixed_6a
        self.Mixed_6b = orig_inception_v3.Mixed_6b
        self.Mixed_6c = orig_inception_v3.Mixed_6c
        self.Mixed_6d = orig_inception_v3.Mixed_6d
        self.Mixed_6e = orig_inception_v3.Mixed_6e
        self.AuxLogits = orig_inception_v3.AuxLogits
        self.Mixed_7a = orig_inception_v3.Mixed_7a
        self.Mixed_7b = orig_inception_v3.Mixed_7b
        self.Mixed_7c = orig_inception_v3.Mixed_7c
        self.avgpool = orig_inception_v3.avgpool
        self.dropout = orig_inception_v3.dropout
        
        self.fc = orig_inception_v3.fc
        self.fc = nn.Linear(self.fc.in_features, self.class_number)

    #def forward(self, x, return_feature_maps=False, avgf=False):
    def forward(self, x, return_feature_maps=False, softmax=False):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        '''
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        '''
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1); features = x.clone()
        # N x 2048
        x = self.fc(x)
        
        if return_feature_maps:
            return features
        
        if softmax:
            normalized_masks = torch.nn.functional.softmax(x, dim=1)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks}
        else:
            return x
        
        
class MobileNetV2_cls(nn.Module):
    def __init__(self, orig_mobilenetV2, class_number=1000):
        super(MobileNetV2_cls, self).__init__()
        
        self.f_dim = 1280
        self.input_dim = 224
        self.class_number = class_number
        
        self.features = orig_mobilenetV2.features
        self.classifier = orig_mobilenetV2.classifier
        self.classifier[1] = nn.Linear(orig_mobilenetV2.classifier[1].in_features, self.class_number)
    

    #def forward(self, x, avgf=False):
    def forward(self, x, return_feature_maps=False, softmax=False):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1); features = x.clone()
        x = self.classifier(x)
    
        if return_feature_maps:
            return features
        
        if softmax:
            normalized_masks = torch.nn.functional.softmax(x, dim=1)
            pred_cls = normalized_masks.argmax(1)
            # _, pred_cls = torch.max(normalized_masks.data, 1)
            return {'class':pred_cls, 'softmax':normalized_masks}
        else:
            return x
        
# parse model args, build and return model
def build_model(arch, class_number, pretrained=True):
    arch = arch.lower()
    if arch == 'resnet18':
        orig_resnet = torchvision.models.resnet18(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'resnet34':
        orig_resnet = torchvision.models.resnet34(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'resnet50':
        orig_resnet = torchvision.models.resnet50(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'resnet101':
        orig_resnet = torchvision.models.resnet101(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'resnet152':
        orig_resnet = torchvision.models.resnet152(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'resnext50_32x4d':
        orig_resnet = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'wide_resnet50_2':
        orig_resnet = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'wide_resnet101_2':
        orig_resnet = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        model = Resnet_cls(orig_resnet, class_number=class_number)
    elif arch == 'hrnet':
        raise NotImplementedError
    elif arch == 'alexnet':
        orig_alexnet = torchvision.models.alexnet(pretrained=pretrained)
        model = AlexNet_cls(orig_alexnet, class_number=class_number)
    elif arch == 'vgg11':
        orig_vgg = torchvision.models.vgg11(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg13':
        orig_vgg = torchvision.models.vgg13(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg16':
        orig_vgg = torchvision.models.vgg16(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg19':
        orig_vgg = torchvision.models.vgg19(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg11_bn':
        orig_vgg = torchvision.models.vgg11_bn(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg13_bn':
        orig_vgg = torchvision.models.vgg13_bn(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg16_bn':
        orig_vgg = torchvision.models.vgg16_bn(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'vgg19_bn':
        orig_vgg = torchvision.models.vgg19_bn(pretrained=pretrained)
        model = VGGs_cls(orig_vgg, class_number=class_number)
    elif arch == 'squeezenet1_0':
        orig_squeezenet = torchvision.models.squeezenet1_0(pretrained=pretrained)
        model = Squeezenets_cls(orig_squeezenet, class_number=class_number)
    elif arch == 'squeezenet1_1':
        orig_squeezenet = torchvision.models.squeezenet1_1(pretrained=pretrained)
        model = Squeezenets_cls(orig_squeezenet, class_number=class_number)
    elif arch == 'densenet121':
        orig_densenet = torchvision.models.densenet121(pretrained=pretrained)
        f_dim = 50176
        model = Densenets_cls(orig_densenet, class_number=class_number)
    elif arch == 'densenet161':
        orig_densenet = torchvision.models.densenet161(pretrained=pretrained)
        f_dim = 108192
        model = Densenets_cls(orig_densenet, class_number=class_number)
    elif arch == 'densenet169':
        orig_densenet = torchvision.models.densenet169(pretrained=pretrained)
        f_dim = 81536
        model = Densenets_cls(orig_densenet, class_number=class_number)
    elif arch == 'densenet201':
        orig_densenet = torchvision.models.densenet201(pretrained=pretrained)
        f_dim = 94080
        model = Densenets_cls(orig_densenet, class_number=class_number)
    elif arch == 'inception_v3':
        orig_inception_v3 = torchvision.models.inception_v3(pretrained=pretrained)
        model = Inception_v3_cls(orig_inception_v3, class_number=class_number)
    elif arch == 'mobilenet_v2':
        orig_mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=pretrained)
        model = MobileNetV2_cls(orig_mobilenet_v2, class_number=class_number)
    elif arch == 'simplemodel':
        # this model is for testing purpose, is it a simple regression model
        model = SimpleModel()
    elif arch == 'mlp':
        model = MLPNet(class_number=class_number)
    else:
        raise Exception('Architecture undefined!')
        
    return model