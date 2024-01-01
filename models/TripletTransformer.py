# PyTorch stuff
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision
from torchvision.models import vision_transformer

avail_backbones = {
    'vit_b_16':vision_transformer.vit_b_16,
    'vit_b_32':vision_transformer.vit_b_32,
    'vit_l_16':vision_transformer.vit_l_16,
    'vit_l_32':vision_transformer.vit_l_32,
    'vit_h_14':vision_transformer.vit_h_14,
}

avail_backbone_urls = {
    'vit_b_16':'https://download.pytorch.org/models/vit_b_16-c867db91.pth',
    'vit_b_32':'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth',
    'vit_l_16':'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth',
    'vit_l_32':'https://download.pytorch.org/models/vit_l_32-c7638314.pth',
    'vit_h_14':'https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth',
}

avail_weights = {
    'vit_b_16':torchvision.models.ViT_B_16_Weights.DEFAULT,
    'vit_b_32':torchvision.models.ViT_B_32_Weights.DEFAULT,
    'vit_l_16':torchvision.models.ViT_L_16_Weights.DEFAULT,
    'vit_l_32':torchvision.models.ViT_L_32_Weights.DEFAULT,
    'vit_h_14':torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1, # DEFAULT require (518 * 518) image size
}

class Triplet_Transformer(nn.Module):
    def __init__(self, backbone, num_classes, embedding_size, pretrained):
        super(Triplet_Transformer, self).__init__()
        self.backbone = self.load_backbone(backbone, pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1000, 500)
        self.fc_embedding = nn.Linear(500, embedding_size)

    def load_backbone(self, backbone, pretrained):
        weights = None if not pretrained else avail_weights[backbone]
        model = avail_backbones[backbone](weights=weights)
        return model

    def forward_sibling(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)

        x_embedding = self.fc_embedding(x)
        return x_embedding

    def forward(self, input1, input2, input3):
        embedding_vec_1 = self.forward_sibling(input1)
        embedding_vec_2 = self.forward_sibling(input2)
        embedding_vec_3 = self.forward_sibling(input3)

        return embedding_vec_1, embedding_vec_2, embedding_vec_3

def TripletTransformer(backbone='vit_b_16', pretrained=False, num_classes=50, embedding_size=128, **kwargs):
    model = Triplet_Transformer(
                                backbone=backbone,
                                num_classes=num_classes,
                                embedding_size=embedding_size,
                                pretrained=pretrained,
                                **kwargs)

    # if pretrained:
    #     # Get imagenet weights for resnet50
    #     weights_imagenet = model_zoo.load_url(model_urls['resnet50'])
    #
    #     # Re-name some of the layers to match our network
    #     weights_imagenet["fc_softmax.weight"] = weights_imagenet["fc.weight"]
    #     weights_imagenet["fc_softmax.bias"] = weights_imagenet["fc.bias"]
    #     weights_imagenet["fc_embedding.weight"] = weights_imagenet["fc.weight"]
    #     weights_imagenet["fc_embedding.bias"] = weights_imagenet["fc.bias"]
    #
    #     # Try and load the model state
    #     model.load_state_dict(weights_imagenet)
    #
    # model.fc = nn.Linear(2048, 1000)
    # model.fc_embedding = nn.Linear(1000, embedding_size)
    # model.fc_softmax = nn.Linear(1000, num_classes)

    return model
