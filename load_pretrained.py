import os
import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary


def load_pretrained():
    # create model
    arch = 'resnet50'
    pretrained = './pretrained/moco_v2_800ep_pretrain.pth.tar'
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch](num_classes=128)

    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            # if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        # args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))

    model = model.to('cuda')
    summary(model, input_size=(3, 224, 224))
    return model


class SubModel(nn.Module):
    def __init__(self, pretrained):
        super(SubModel, self).__init__()
        image_modules = list(pretrained.children())[:-1]  # all layer expect last layer
        self.modelA = nn.Sequential(*image_modules)

    def forward(self, inputs):
        x = self.modelA(inputs)
        x = torch.flatten(x, 1)
        return x


def main():
    import numpy as np

    model = load_pretrained()
    sub_model = SubModel(model)
    sub_model.eval()

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    image_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    image_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]

    test_inputs = torch.ones(size=(1, 3, 224, 224), dtype=torch.float32) * 0.5
    test_inputs = (test_inputs - image_mean) / image_std
    test_inputs = test_inputs.to('cuda')
    pytorch_out = sub_model(test_inputs)
    print(pytorch_out.size())

    np.save('pytorch_out.npy', pytorch_out.cpu().detach().numpy())
    # print(pytorch_out.cpu().detach().numpy())
    return


if __name__ == '__main__':
    main()
