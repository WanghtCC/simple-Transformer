import torch
import torch.nn as nn
from einops import rearrange

from base_trans.transformer_base import transformer_base


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer_base(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, mask=None):
        p = self.patch_size
        print("init", x.shape)

        x = rearrange(
            x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p
        )  # [H W C] -> [N PPC]
        print(x.shape, "rearrange")
        x = self.patch_to_embedding(x)  # [N PPC] -> [N PPC D]
        print(x.shape, "patch_to_embedding")
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [N 1 D]
        print(cls_tokens.shape, "cls_tokens")
        x = torch.cat((cls_tokens, x), dim=1)  # [N PPC+1 D]
        print(x.shape, "cat")
        print(self.pos_embedding.shape, "pos_embedding")
        x = x + self.pos_embedding
        x = self.transformer(x, mask)
        print(x.shape, "after transformer")
        x = self.to_cls_token(x[:, 0])
        print(x.shape, "after to_cls_token")
        x = self.mlp_head(x)
        print(x.shape, "after mlp_head")
        return x


if __name__ == "__main__":
    model = ViT(
        image_size=28,
        patch_size=4,
        num_classes=10,
        dim=64,
        depth=6,
        heads=8,
        mlp_dim=128,
        channels=1,
    )

    x = torch.randn(1, 1, 28, 28)
    print(model(x).shape)
    print(model)
