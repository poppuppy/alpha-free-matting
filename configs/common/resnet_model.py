from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import ResNet, BasicStem
from modeling import SegMatte, SegmentationCriterion, LCADecoder


model = L(SegMatte)(
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=True,
            norm="FrozenBN",
        ),
        out_features=["res2", "res3", "res4", "res5"],
        # freeze_at=1,
    ),
    decoder=L(LCADecoder)(
        in_features=["res2", "res3", "res4", "res5"],
        in_channels=[256, 512, 1024, 2048]
    ),
    criterion=L(SegmentationCriterion)(
        losses=['loss_binary_cross_entropy']
    ),
    pixel_mean=[123.675 / 255., 116.280 / 255., 103.530 / 255.],
    pixel_std=[58.395 / 255., 57.120 / 255., 57.375 / 255.],
    input_format="RGB",
    size_divisibility=32,
)
