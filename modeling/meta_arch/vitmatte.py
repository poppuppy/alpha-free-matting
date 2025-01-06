import torch
import torch.nn as nn


class ViTMatte(nn.Module):
    def __init__(self,
                 *,
                 backbone,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.backbone = backbone
        self.criterion = criterion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, targets, H, W = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images)
        outputs = self.decoder(features, images)

        if self.training:
            assert targets is not None
            losses = self.criterion(outputs, targets)
            return losses
        else:
            outputs['phas'] = outputs['phas'][:, :, :H, :W]
            return outputs

    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        self.image = images[-1, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        images = (images - self.pixel_mean) / self.pixel_std

        targets = dict()

        if self.training:
            trimap = batched_inputs['trimap'].to(self.device)

            # trimap = trimap.float() / 255.
            trimap = trimap.float()
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

            targets.update(dict(trimap=trimap))

        B, C, H, W = images.shape
        if images.shape[-1] % 32 != 0 or images.shape[-2] % 32 != 0:
            new_H = (32 - images.shape[-2] % 32) + H
            new_W = (32 - images.shape[-1] % 32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:, :, :H, :W] = images[:, :, :, :]
            del images
            images = new_images

        return images, targets, H, W
