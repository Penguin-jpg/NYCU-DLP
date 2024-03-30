from torchvision.models import vgg19_bn

vgg = vgg19_bn(pretrained=False)
print(vgg)
