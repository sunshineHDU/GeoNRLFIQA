import torch
import timm
from torch import nn
from einops import rearrange
from models.GeoLearning import GeoLearning
from config import config


class GeoLFIQA(nn.Module):
    def __init__(self, embed_dim=72, patch_size=8, drop=0.1, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size

        self.vit = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)

        self.feature_num = 512
        self.out_num = 256

        self.conv1 = nn.Conv2d(embed_dim * 3, self.feature_num, 1, 1, 0)
        self.conv2 = nn.Conv2d(self.feature_num, self.out_num, 1, 1, 0)

        self.fc_1 = nn.Sequential(
            nn.Linear(self.out_num, self.out_num),
            nn.GELU(),
            nn.Linear(self.out_num, 1),
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(self.out_num, self.out_num),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(self.out_num, 1),
        )

        self.GeoLearning1 = GeoLearning(
                        image_size=img_size,
                        patch_size=16,
                        dim=self.feature_num,
                        depth=6,
                        heads=16,
                        mlp_dim=2048,
                        dropout=0.1,
                        emb_dropout=0.1,
                    )

        self.GeoLearning2 = GeoLearning(
                        image_size=img_size,
                        patch_size=16,
                        dim=self.out_num,
                        depth=6,
                        heads=16,
                        mlp_dim=2048,
                        dropout=0.1,
                        emb_dropout=0.1,
                    )

    def forward(self, x):
        x0 = self.vit(x[0]).cuda()
        x1 = self.vit(x[1]).cuda()
        x2 = self.vit(x[2]).cuda()

        x = torch.cat((x0, x1, x2), dim=2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_size, w=self.input_size)

        x = self.conv1(x)
        x = self.GeoLearning1(x)
        x = self.conv2(x)
        x = self.GeoLearning2(x)

        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', h=h, w=w)
        x = x[:, 0, :]

        if config.db_name  == 'win5':
            score = self.fc_1(x)
        else:
            score  =  self.fc_2(x)

        return score
