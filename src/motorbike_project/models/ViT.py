import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 384,
        num_layers: int = 6,
        img_size: int = 112,
        num_heads: int = 6,
        num_patches: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        assert img_size % num_patches == 0, 'Image size must be divisible by number of patches'
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches**2+1, embed_dim))
        self.linear_embed = nn.Conv2d(in_channels=3,  # Modified in_channels to match input channels
                                      out_channels=embed_dim,
                                      kernel_size=img_size // num_patches,
                                      stride=img_size // num_patches,
                                      padding=0)
        self.dropout = nn.Dropout(dropout)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                       nhead=num_heads,
                                                       dim_feedforward=embed_dim*4,
                                                       dropout=dropout,
                                                       batch_first=True)

        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.linear_embed(x).flatten(2).permute(0, 2, 1)  # B 16 256
        x = torch.cat(
            [self.cls_token.expand(x.shape[0], -1, -1), x],
            dim=1
        )  # B 17 256
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.dropout2(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    vit = VisionTransformer(num_layers=3)
    x = torch.randn(2, 3, 112, 112)
    y = vit(x)
    print(y)
