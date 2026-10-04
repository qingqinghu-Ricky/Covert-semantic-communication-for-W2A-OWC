"""Original Fig. 7 architecture; layer names retained for checkpoint compatibility."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class ResNet50FPN(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.backbone = resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        self.lateral_c2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.lateral_c3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.lateral_c4 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.lateral_c5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fpn_p2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fpn_p3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fpn_p4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fpn_p5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def fpn_forward(self, c2, c3, c4, c5):
        lat_c2 = self.lateral_c2(c2)
        lat_c3 = self.lateral_c3(c3)
        lat_c4 = self.lateral_c4(c4)
        lat_c5 = self.lateral_c5(c5)
        p5 = self.fpn_p5(lat_c5)
        p4 = self.fpn_p4(
            F.interpolate(
                p5, size=lat_c4.shape[2:], mode="bilinear", align_corners=True
            )
            + lat_c4
        )
        p3 = self.fpn_p3(
            F.interpolate(
                p4, size=lat_c3.shape[2:], mode="bilinear", align_corners=True
            )
            + lat_c3
        )
        p2 = self.fpn_p2(
            F.interpolate(
                p3, size=lat_c2.shape[2:], mode="bilinear", align_corners=True
            )
            + lat_c2
        )
        return p2

    def forward(self, x):
        orig_size = x.shape[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p2 = self.fpn_forward(c2, c3, c4, c5)
        p2_up = F.interpolate(p2, size=orig_size, mode="bilinear", align_corners=True)
        output = self.seg_head(p2_up)
        return output


class FPN(nn.Module):
    def __init__(self, in_channels_list=[128, 256, 512], out_channel=512):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_c, out_channel, kernel_size=1, stride=1, padding=0)
                for in_c in in_channels_list
            ]
        )
        self.smooth_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
                for _ in in_channels_list
            ]
        )
        self.out_channel = out_channel

    def forward(self, features):
        lateral_feats = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        fused_feats = [lateral_feats[-1]]
        for i in range(len(lateral_feats) - 2, -1, -1):
            upsampled = nn.functional.interpolate(
                fused_feats[-1],
                size=lateral_feats[i].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = upsampled + lateral_feats[i]
            smoothed = self.smooth_convs[i](fused)
            fused_feats.append(smoothed)
        return fused_feats[-1]


class Semantic_Encoder(nn.Module):
    def __init__(
        self, num_classes=6, latent_dim=1024, input_h=320, input_w=256, pretrained=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.input_h = input_h
        self.input_w = input_w
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained is True else None
        )
        if isinstance(pretrained, (str, bytes)):
            self.backbone.load_state_dict(
                torch.load(pretrained, map_location="cpu", weights_only=True),
                strict=True,
            )
        self.backbone.conv1 = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.stage1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.stage2 = self.backbone.layer1
        self.stage3 = self.backbone.layer2
        self.stage4 = self.backbone.layer3
        self.stage5 = self.backbone.layer4
        self.fpn = FPN(in_channels_list=[128, 256, 512], out_channel=256)
        self.final_c, self.final_h, self.final_w = self._compute_final_feature_size()
        self.conv_downsampler = nn.Sequential(
            nn.Conv2d(
                self.final_c // 2,
                self.final_c // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.final_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.final_c // 2,
                self.final_c // 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.final_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.final_c // 2,
                out_channels=self.final_c,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.BatchNorm2d(self.final_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.final_c,
                out_channels=self.latent_dim,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
        )

    def _compute_final_feature_size(self):
        final_c = 512
        downsample_count = 5
        final_h = self.input_h // 2**downsample_count
        final_w = self.input_w // 2**downsample_count
        assert (
            self.input_h % 2**downsample_count == 0
        ), f"input_h={self.input_h} 无法被 {2 ** downsample_count} 整除"
        assert (
            self.input_w % 2**downsample_count == 0
        ), f"input_w={self.input_w} 无法被 {2 ** downsample_count} 整除"
        return (final_c, final_h, final_w)

    def forward(self, x):
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        fpn_feat = self.fpn([feat3, feat4, feat5])
        features_downsamp = self.conv_downsampler(fpn_feat)
        latent = features_downsamp.flatten(1)
        return latent


class Semantic_Decoder(nn.Module):
    def __init__(
        self,
        num_classes=6,
        latent_dim=1024,
        encoder_final_c=512,
        encoder_final_h=10,
        encoder_final_w=8,
        output_h=320,
        output_w=256,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.encoder_final_c = encoder_final_c
        self.encoder_final_h = encoder_final_h
        self.encoder_final_w = encoder_final_w
        self.output_h = output_h
        self.output_w = output_w
        self.conv_upsampler = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_dim,
                self.encoder_final_c * 2,
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.encoder_final_c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                self.encoder_final_c * 2,
                self.encoder_final_c,
                kernel_size=3,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.encoder_final_c),
            nn.ReLU(inplace=True),
        )
        self.deconv_blocks = self._build_deconv_blocks()

    def _build_deconv_blocks(self):
        layers = []
        in_channels = self.encoder_final_c
        upsample_configs = [(256, 2), (128, 2), (64, 2), (32, 2), (self.num_classes, 2)]
        for out_channels, scale in upsample_configs:
            layers.extend(
                [
                    nn.Upsample(
                        scale_factor=scale, mode="bilinear", align_corners=False
                    ),
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels)
                    if out_channels != self.num_classes
                    else nn.Identity(),
                    nn.ReLU(inplace=True)
                    if out_channels != self.num_classes
                    else nn.Identity(),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, features):
        features_v1 = features.unsqueeze(2).unsqueeze(3)
        features_upsamp = self.conv_upsampler(features_v1)
        output = self.deconv_blocks(features_upsamp)
        return output


class SparseMaskGenerator(nn.Module):
    def __init__(self, latent_dim, img_feat_dim=32, snr_feat_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_mask = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Linear(latent_dim // 4, img_feat_dim),
            nn.ReLU(),
        )
        self.snr_mask = nn.Sequential(
            nn.Linear(1, snr_feat_dim),
            nn.ReLU(),
            nn.Linear(snr_feat_dim, snr_feat_dim),
            nn.ReLU(),
        )
        self.fusion_layers = nn.Sequential(
            nn.Linear(img_feat_dim + snr_feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
            nn.Sigmoid(),
        )

    def forward(self, latent, snr):
        data_feat = self.data_mask(latent)
        snr_feat = self.snr_mask(snr)
        fusion_feat = torch.cat([data_feat, snr_feat], dim=1)
        sparse_mask = self.fusion_layers(fusion_feat)
        return sparse_mask
