import torch
import torch.nn as nn
import torch.nn.functional as F


class MattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 losses,
                 aux_losses=None,
                 ):
        super(MattingCriterion, self).__init__()
        self.losses = losses
        self.aux_losses = aux_losses

    def loss_gradient_penalty(self, sample_map, preds, targets):
        preds = preds['phas']
        targets = targets['phas']

        #sample_map for unknown area
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        
        scale = sample_map.shape[0]*512*512/torch.sum(sample_map)

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, outputs, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*512*512/torch.sum(new_sample_map)

        # loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        loss = F.l1_loss(outputs['phas']*new_sample_map, targets['trimap']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def l1_loss(self, sample_map, outputs, targets):
        loss = F.l1_loss(outputs['phas'], targets['trimap'])
        return dict(l1_loss=loss)

    def consistency_loss(self, sample_map, outputs, targets):
        loss = F.l1_loss(outputs['images_dist'], outputs['phas_dist']) * 10.
        return dict(consistency_loss=loss)

    def get_loss(self, loss, sample_map, outputs, targets):
        loss_map = {
            'l1': self.l1_loss,
            'known': self.known_l1_loss,
            'consistency': self.consistency_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](sample_map, outputs, targets)

    def forward(self, outputs, targets):
        sample_map = torch.zeros_like(targets['trimap'])
        sample_map[targets['trimap'] == 0.5] = 1  # 0.5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        losses = dict()
        for loss in self.losses:
            losses.update(self.get_loss(loss, sample_map, outputs, targets))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # sample_map = torch.zeros_like(aux_targets['trimap'])
                # sample_map[aux_targets['trimap'] == 0.5] = 1
                for loss in self.aux_losses:
                    l_dict = self.get_loss(loss, sample_map, aux_outputs, targets)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]