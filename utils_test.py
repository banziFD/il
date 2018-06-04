import numpy as np
from PIL import Image
import torch

def show_protoset(protoset, cl):
    if(not any(protoset)):
        print('Empty protoset!')
        return
    keys = protoset.keys()
    if(cl not in keys):
        print('Not a recognisable class!')
        return
    imgs = protoset[cl][1]
    imgs = imgs.transpose([0, 2, 3, 1])
    shape = imgs.shape
    print(shape)
    nb_proto = shape[0]
    x_pix = shape[1]
    y_pix = shape[2]
    x_grid = (int)(nb_proto ** 0.5)
    y_grid = (int)(nb_proto / x_grid)
    if(nb_proto > x_grid * y_grid):
        y_grid += 1
    result = Image.new('RGB', (x_grid * x_pix, y_grid * y_pix))
    for i in range(shape[0]):
        x = (int)(i / y_grid)
        y = (int)(i % y_grid)
        img = imgs[i]
        img = Image.fromarray(img, mode = 'RGB')
        result.paste(img, (x * x_pix, y * y_pix))
    result.show()

if __name__ == '__main__':
    a = np.random.randint(256, size = (11, 3, 32, 32), dtype = 'uint8')
    a = {4:([], a, [])}
    show_protoset(a, 4)