import random
import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    """
    Randomly flip along `z`-, `y`- and `x`-axes as well as swap `y`- and `x`-axes 
    for anisotropic image volumes. For learning on isotropic image volumes set 
    :attr:`do_ztrans` to 1 to swap `z`- and `x`-axes (the inputs need to be cubic).

    Args:
        p (float): probability of applying the augmentation. Default: 0.5
        do_ztrans (int): set to 1 to swap z- and x-axes for isotropic data. Default: 0
    """
    def __init__(self, p=0.5, do_ztrans=0):
        super(Flip, self).__init__(p)
        self.do_ztrans = do_ztrans

    def set_params(self):
        # No change in sample size
        pass

    def flip_and_swap(self, data, rule):
        assert data.ndim==3 or data.ndim==4
        if data.ndim == 3: # 3-channel input in z,y,x
            # z reflection.
            if rule[0]:
                data = data[::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 2, 1)
            # Transpose in xz.
            if self.do_ztrans==1 and rule[4]:
                data = data.transpose(2, 1, 0)
        else: # 4-channel input in c,z,y,x
            # z reflection.
            if rule[0]:
                data = data[:, ::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, :, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 1, 3, 2)
            # Transpose in xz.
            if self.do_ztrans==1 and rule[4]:
                data = data.transpose(0, 3, 2, 1)
        return data
    
    def __call__(self, data, random_state=np.random):
        output = {}

        rule = random_state.randint(2, size=4+self.do_ztrans)
        augmented_image = self.flip_and_swap(data['image'], rule)
        augmented_label = self.flip_and_swap(data['label'], rule)
        output['image'] = augmented_image
        output['label'] = augmented_label
        if 'distance' in data:
            augmented_distance = self.flip_and_swap(data['distance'], rule)
            output['distance'] = augmented_distance
        if 'feature' in data:
            augmented_feature = self.flip_and_swap(data['feature'], rule)
            output['feature'] = augmented_feature
        if 'embedding' in data:
            augmented_embedding = self.flip_and_swap(data['embedding'], rule)
            output['embedding'] = augmented_embedding
        if 'flow' in data:
            augmented_flow = self.flip_and_swap(data['flow'], rule)
            if rule[0]:
                augmented_flow *= -1
                augmented_flow_shift = np.zeros_like(augmented_flow)
                augmented_flow_shift[:-1] = augmented_flow[1:]
                augmented_flow_shift[-1] = augmented_flow[0]
                augmented_flow = augmented_flow_shift
            output['flow'] = augmented_flow

        return output



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    import imageio

    stack = imageio.volread('/mnt/WGCJ/SAM/sofima/inputs/SNEMI.tif')
    z_s = np.random.randint(0, stack.shape[0]-20)
    y_s = np.random.randint(0, stack.shape[1]-240)
    x_s = np.random.randint(0, stack.shape[2]-240)
    im = stack[z_s:z_s+20, y_s:y_s+240, x_s:x_s+240]

    # im = np.random.randn(20, 240, 240)
    im = np.transpose(im, (1,2,0))
    im_x4 = cv2.resize(im, (0,0), fx=1/4, fy=1/4, interpolation=cv2.INTER_LINEAR)
    im_x4 = np.transpose(im_x4, (2,0,1))
    im_x16 = cv2.resize(im, (0,0), fx=1/16, fy=1/16, interpolation=cv2.INTER_LINEAR)
    im_x16 = np.transpose(im_x16, (2,0,1))
    im = np.transpose(im, (2,0,1))
    
    data = {'image': im, 'label': im, 
            'feature': np.repeat(im_x4[np.newaxis, ...], 32, axis=0), 
            'embedding': np.repeat(im_x16[np.newaxis, ...], 256, axis=0),
            'flow': np.repeat(im_x4[np.newaxis, ...], 2, axis=0).astype(np.float32)}
    print(data['image'].shape, data['feature'].shape, data['embedding'].shape, data['flow'].shape)
    flip = Flip(p=1.0, do_ztrans=0)

    fig, ax = plt.subplots(10, 4, figsize=(12, 26))
    for i in range(10):
        data_aug = flip(data)
        ax[i,0].imshow(data_aug['image'][10], cmap='gray')
        ax[i,1].imshow(data_aug['feature'][0, 10], cmap='gray')
        ax[i,2].imshow(data_aug['embedding'][0, 10], cmap='gray')
        ax[i,3].imshow(data_aug['flow'][0, 10], cmap='gray')
    plt.savefig('test_flip.png')



'''
change "from .augmentor import DataAugment" to "from augmentor import DataAugment"

/mnt/WGCJ/miniconda4/envs/micro-sam/bin/python flip.py
'''