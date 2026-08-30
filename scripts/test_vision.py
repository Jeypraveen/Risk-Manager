import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

p = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
m = AutoModel.from_pretrained('facebook/dinov2-small')

def prep(p_path):
    im = Image.open(p_path)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        im = im.convert('RGBA')
        bg = Image.new('RGB', im.size, (255,255,255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert('RGB')

im1 = prep('data/images/catalog/smartphone.png')
im2 = prep('data/images/returns/smartphone_legitimate.png')

o1 = m(**p(im1, return_tensors='pt'))
o2 = m(**p(im2, return_tensors='pt'))

e1 = o1.last_hidden_state[:,0,:].detach().numpy().flatten()
e2 = o2.last_hidden_state[:,0,:].detach().numpy().flatten()

print('White BG CLS:', np.dot(e1, e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)))
