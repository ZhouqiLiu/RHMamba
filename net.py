
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from LSConv import LSConv, SpatialGate
from refiner import HardPointRefinerV3
from ah_ssm import VSSBlock


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def weight_init(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, val=0.0)

    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, val=1.0)
        nn.init.constant_(layer.bias, val=0.0)

    elif isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, val=0.0)


class HetConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        bias=None,
        p=64,
        g=64,
    ):
        super().__init__()

        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        max_common_divisor = gcd(in_channels, out_channels)

        valid_groups = [
            i
            for i in range(1, max_common_divisor + 1)
            if in_channels % i == 0 and out_channels % i == 0
        ]

        if g in valid_groups:
            actual_g = g
        else:
            actual_g = min(
                valid_groups,
                key=lambda x: abs(x - g),
            )


        self.gwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            groups=actual_g,
            padding=kernel_size // 3,
            stride=stride,
        )

        self.pwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=p,
            stride=stride,
        )

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)


class Network(nn.Module):
    def __init__(
        self,
        FM,
        NC,
        NCLidar,
        Classes,
        patchsize,
        num_token,
        use_lska=True,
        use_sdd=True,
        ssm_scan_mode="hybrid",
        lambda_reg=0.05,
        vss_depth=1,
    ):
        super().__init__()

        if not use_lska or not use_sdd:
            raise ValueError(
                "Require: use_lska=True、use_sdd=True。"
            )
        if ssm_scan_mode != "hybrid":
            raise ValueError(
                "Require: ssm_scan_mode='hybrid'。"
            )
        if vss_depth != 1:
            raise ValueError(
                "Require:1 VSSBlock。"
            )

        self.FM = FM
        self.NC = NC
        self.NCLidar = NCLidar
        self.patchsize = patchsize
        self.num_token = num_token
        self.use_lska = use_lska
        self.use_sdd = use_sdd
        self.ssm_scan_mode = ssm_scan_mode

        self.lambda_reg = float(lambda_reg)
        self.vss_depth = vss_depth

        self.conv5 = nn.Sequential(
            nn.Conv3d(
                1,
                8,
                (9, 3, 3),
                padding=(0, 1, 1),
                stride=1,
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            HetConv(
                8 * (NC - 8),
                FM * 4,
                p=1,
                g=(
                    (FM * 4) // 4
                    if (8 * (NC - 8)) % FM == 0
                    else (FM * 4) // 8
                ),
            ),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
        )

        self.lidarConv = nn.Sequential(
            nn.Conv2d(NCLidar, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.GELU(),
        )

        self.lsconv = LSConv(
            dim=FM * 4,
            out_channels=FM * 4,
        )
        self.spatial_gate = SpatialGate(channels=FM * 4)

        self.projection = nn.Sequential(
            nn.Conv1d(2, 1, 1, 1, 0),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )

        self.token_HSIwA_L = nn.Parameter(
            torch.empty(1, 1, FM * 4),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.token_HSIwA_L)

        self.token_HSIwV_L = nn.Parameter(
            torch.empty(1, FM * 4, FM * 4),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.token_HSIwV_L)

        self.xishu1 = nn.Parameter(
            torch.Tensor([0.5]),
            requires_grad=True,
        )

        self.expand_layer = nn.Linear(
            num_token + 1,
            self.patchsize * self.patchsize,
        )

        self.refiner = HardPointRefinerV3(
            in_channels=FM * 4,
            thresh=0.6,
            temp=8.0,
            share_alpha=False,
            use_reg=True,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.vssblock1 = VSSBlock(
            dim=FM * 4,
            drop_path=0.1,
            d_state=16,
            mlp_ratio=2.0,
            ssm_scan_mode=self.ssm_scan_mode,
        )


        self.out3 = nn.Linear(FM * 4, Classes)

        self.position_embeddings = nn.Parameter(
            torch.randn(
                1,
                self.num_token + 1,
                FM * 4,
            )
        )
        self.dropout = nn.Dropout(0.1)

        nn.init.xavier_uniform_(self.out3.weight)
        nn.init.normal_(self.out3.bias, std=1e-6)

        self.token_wA = nn.Parameter(
            torch.empty(1, self.num_token, FM * 4),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.token_wA)

        self.token_wV = nn.Parameter(
            torch.empty(1, FM * 4, FM * 4),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.token_wV)

        self.token_wA_L = nn.Parameter(
            torch.empty(1, 1, FM * 4),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.token_wA_L)

        self.token_wV_L = nn.Parameter(
            torch.empty(1, FM * 4, FM * 4),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.token_wV_L)

    def masked_classification_loss(self, pred, hard_mask):
        mask_weight = hard_mask.mean(dim=[1, 2, 3])

        with torch.no_grad():
            probs = F.softmax(pred, dim=-1)
            conf = probs.max(dim=-1)[0]

        return torch.mean(mask_weight * (1.0 - conf))

    def forward(self, x1, x2):

        x1 = x1.unsqueeze(1)
        x1 = self.conv5(x1)
        x1 = x1.reshape(
            x1.shape[0],
            -1,
            self.patchsize,
            self.patchsize,
        )
        x1 = self.conv6(x1)


        x2 = self.lidarConv(x2)
        x2 = self.lsconv(x2)
        x1, x2 = self.spatial_gate(x1, x2)


        x1, x2, bad_point, reg_loss = self.refiner(x1, x2)


        x2 = x2.reshape(
            x2.shape[0],
            -1,
            self.patchsize ** 2,
        )
        x2 = x2.transpose(-1, -2)

        wa_L = self.token_wA_L.expand(x1.shape[0], -1, -1)
        wa_L = rearrange(wa_L, "b h w -> b w h")

        A_L = torch.einsum("bij,bjk->bik", x2, wa_L)
        A_L = rearrange(A_L, "b h w -> b w h")
        A_L = A_L.softmax(dim=-1)

        wv_L = self.token_wV_L.expand(x2.shape[0], -1, -1)
        VV_L = torch.einsum("bij,bjk->bik", x2, wv_L)
        x2 = torch.einsum("bij,bjk->bik", A_L, VV_L)

        x1 = x1.flatten(2)
        x1 = x1.transpose(-1, -2)

        HSIwa_L = self.token_HSIwA_L.expand(
            x1.shape[0], -1, -1
        )
        HSIwa_L = rearrange(HSIwa_L, "b h w -> b w h")

        HSI_L = torch.einsum("bij,bjk->bik", x1, HSIwa_L)
        HSI_L = rearrange(HSI_L, "b h w -> b w h")
        HSI_L = HSI_L.softmax(dim=-1)

        HSIwv_L = self.token_HSIwV_L.expand(
            x2.shape[0], -1, -1
        )
        HSI_VL = torch.einsum("bij,bjk->bik", x1, HSIwv_L)
        HSI_token = torch.einsum(
            "bij,bjk->bik", HSI_L, HSI_VL
        )

        x2 = (
            self.xishu1 * x2
            + (1 - self.xishu1) * HSI_token
        )
        x2 = torch.cat((x2, HSI_token), dim=1)


        x2 = self.projection(x2)


        wa = self.token_wA.expand(x1.shape[0], -1, -1)
        wa = rearrange(wa, "b h w -> b w h")

        A = torch.einsum("bij,bjk->bik", x1, wa)
        A = rearrange(A, "b h w -> b w h")
        A = A.softmax(dim=-1)

        wv = self.token_wV.expand(x1.shape[0], -1, -1)
        VV = torch.einsum("bij,bjk->bik", x1, wv)
        T = torch.einsum("bij,bjk->bik", A, VV)

        x = torch.cat((x2, T), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        embeddings = self.expand_layer(
            embeddings.transpose(1, 2)
        )
        embeddings = embeddings.view(
            embeddings.shape[0],
            self.FM * 4,
            self.patchsize,
            self.patchsize,
        )

        x = self.vssblock1(embeddings)
        x = self.pool(x)
        x = x.flatten(1)

        processed_feature = x
        out3 = self.out3(x)

        mask_loss = self.masked_classification_loss(
            out3,
            bad_point,
        )
        if reg_loss is not None:
            mask_loss = mask_loss + self.lambda_reg * reg_loss

        return out3, mask_loss, processed_feature

