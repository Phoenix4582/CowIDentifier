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

class transformer_embeddings(nn.Module):
    def __init__(self, backbone, num_classes, embedding_size, pretrained):
        super(transformer_embeddings, self).__init__()
        self.backbone = self.load_backbone(backbone, pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1000, 500)
        self.fc_softmax = nn.Linear(500, num_classes)
        self.fc_embedding = nn.Linear(500, embedding_size)

    def load_backbone(self, backbone, pretrained):
        weights = None if not pretrained else avail_weights[backbone]
        model = avail_backbones[backbone](weights=weights)
        return model

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.relu(x)

        x_embedding = self.fc_embedding(x)
        return x_embedding

# return a selected Vision Transformer
def ViT(backbone, pretrained=False, num_classes=50, ckpt_path=False, embedding_size=128, **kwargs):
    # Define the model
    model = transformer_embeddings(backbone=backbone,
                                   pretrained=pretrained,
                                   num_classes=num_classes,
                                   embedding_size=embedding_size, **kwargs)

    # Should we load save model weights
    if pretrained:
        # if lightning_mode:
        #     weights_init = torch.load(ckpt_path)['state_dict']
        #     for key in list(weights_init.keys()):
        #         weights_init[key.replace('models.', '')] = weights_init.pop(key)
        weights_init = torch.load(ckpt_path)['model_state']
        model.load_state_dict(weights_init, strict=False)

    return model
