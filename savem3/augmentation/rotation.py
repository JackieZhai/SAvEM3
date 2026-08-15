import cv2
import numpy as np
from .augmentor import DataAugment

class Rotate(DataAugment):
    """
    Continuous rotatation of the `xy`-plane.

    The sample size for `x`- and `y`-axes should be at least :math:`\sqrt{2}` times larger
    than the input size to make sure there is no non-valid region after center-crop.
    
    Args:
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, p=0.5):
        super(Rotate, self).__init__(p=p) 
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.set_params()

    def set_params(self):
        # sqrt(2)
        self.sample_params['ratio'] = [1.0, 1.42, 1.42]

    def rotate(self, imgs, M, interpolation):
        height, width = imgs.shape[-2:]
        transformedimgs = np.copy(imgs)
        for z in range(transformedimgs.shape[-3]):
            img = transformedimgs[z, :, :]
            dst = cv2.warpAffine(img, M ,(height,width), 1.0, flags=interpolation, borderMode=self.border_mode)
            transformedimgs[z, :, :] = dst

        return transformedimgs

    def __call__(self, data, random_state=np.random):

        if 'label' in data and data['label'] is not None:
            image, label = data['image'], data['label']
        else:
            image, label = data['image'], None

        height, width = image.shape[-2:]
        random_angle = random_state.rand()*360.0
        M = cv2.getRotationMatrix2D((height/2, width/2), random_angle, 1)
        M_x4 = cv2.getRotationMatrix2D((height/4/2, width/4/2), random_angle, 1)
        M_x16 = cv2.getRotationMatrix2D((height/16/2, width/16/2), random_angle, 1)

        output = {}
        output['image'] = self.rotate(image, M, self.image_interpolation)
        if label is not None:
            output['label'] = self.rotate(label, M, self.label_interpolation)
        if 'feature' in data:
            output['feature'] = np.zeros_like(data['feature'])
            for c in range(data['feature'].shape[0]):
                output['feature'][c, :] = self.rotate(data['feature'][c, :], M_x4, self.image_interpolation)
        if 'embedding' in data:
            output['embedding'] = np.zeros_like(data['embedding'])
            for c in range(data['embedding'].shape[0]):
                output['embedding'][c, :] = self.rotate(data['embedding'][c, :], M_x16, self.image_interpolation)
        if 'distance' in data:
            output['distance'] = self.rotate(data['distance'], M, self.label_interpolation)
        if 'flow' in data:
            output['flow'] = np.zeros_like(data['flow'])
            for c in range(data['flow'].shape[0]):
                output['flow'][c, :] = self.rotate(data['flow'][c, :], M_x4, self.image_interpolation)

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
    rotate = Rotate(p=0.5)

    fig, ax = plt.subplots(10, 4, figsize=(12, 26))
    for i in range(10):
        data_aug = rotate(data)
        ax[i,0].imshow(data_aug['image'][10], cmap='gray')
        ax[i,1].imshow(data_aug['feature'][0, 10], cmap='gray')
        ax[i,2].imshow(data_aug['embedding'][0, 10], cmap='gray')
        ax[i,3].imshow(data_aug['flow'][0, 10], cmap='gray')
    plt.savefig('test_rotation.png')



'''
change "from .augmentor import DataAugment" to "from augmentor import DataAugment"

/mnt/WGCJ/miniconda4/envs/micro-sam/bin/python rotation.py
'''