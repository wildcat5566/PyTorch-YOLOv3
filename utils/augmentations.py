import torch
import torch.nn.functional as F
import numpy as np

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def random_zoom(images, labels):
    #input tensor 3*H*W
    _, H, W = images.shape
    zoom_ratio = np.random.uniform(0.6, 1)
    borders = (1-zoom_ratio)*0.5
    new_images = images[:, int(borders*H) : H - int(borders*H), int(borders*W) : W - int(borders*W)]
    new_images = F.interpolate(new_images.unsqueeze(dim=0), size=(H, W), mode='bilinear').squeeze()

    cropped_off = False #return original image instead if targets are cropped off
    new_labels = np.zeros_like(labels, dtype=float)
    for i, (l, new_l) in enumerate(zip(labels, new_labels)):
        #boundary issues
        x_offset, y_offset = l[2] - 0.5, l[3] - 0.5
        x = (0.5 + x_offset / zoom_ratio)
        y = (0.5 + y_offset / zoom_ratio)
        w = l[4] / zoom_ratio
        h = l[5] / zoom_ratio

        if (x <= 0 or y <= 0 or x >= 1 or y >= 1):
            cropped_off = True
            return images, labels

        if x + w/2 >= 1.0:
            boxleft = x - w/2
            boxright = 1.0
            x = 0.5*(boxleft + boxright)
            w = boxright - boxleft

        elif x - w/2 < 0:
            boxleft = 0
            boxright = x + w/2
            x = 0.5*(boxleft + boxright)
            w = boxright - boxleft

        if y + h/2 >= 1.0:
            boxup = y - h/2
            boxdown = 1.0
            y = 0.5*(boxup + boxdown)
            h = boxdown - boxup

        elif y - h/2 < 0:
            boxup = 0
            boxdown = y + h/2
            y = 0.5*(boxup + boxdown)
            h = boxdown - boxup

        new_l[:2] = l[:2]
        new_l[2:] = x, y, w, h

    return new_images, torch.Tensor(new_labels)
