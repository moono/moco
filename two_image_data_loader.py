import os
import glob
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.datasets.folder as folder

from torch.utils.data import Dataset
from moco.loader import GaussianBlur


# ref
# https://discuss.pytorch.org/t/getting-the-crop-parameters-used-in-randomresizedcrop/42279/2
# https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914
class CustomDataset(Dataset):
    def __init__(self, image_dir_a, image_dir_b):
        # assert 'customize_data_v9_55000_openface_aligned' in image_dir_a
        # assert 'customize_data_v9_55000_config' in image_dir_b
        self.image_fns_a = sorted(glob.glob(os.path.join(image_dir_a, '*.png')))
        self.image_fns_b = sorted(glob.glob(os.path.join(image_dir_b, '*.png')))
        assert len(self.image_fns_a) == len(self.image_fns_b)

        self.loader = folder.default_loader
        self.final_size = 224
        self.resize_scale = (0.7, 1.0)
        self.resize_ratio = (0.75, 1.33)
        self.p_color_jitter = 0.8
        self.color_jitter_param = (0.4, 0.4, 0.4, 0.1)
        self.p_grayscale = 0.2
        self.p_gaussian_blur = 0.5
        self.p_horizontal_flip = 0.5

        # image mean / std
        self.mean_a = [0.70263284, 0.6189805, 0.5885812]
        self.std_a = [0.17044774, 0.19329663, 0.2288487]
        self.mean_b = [0.73953557, 0.6567175, 0.6033062]
        self.std_b = [0.16435608, 0.19073875, 0.2246457]
        return

    def __len__(self):
        return len(self.image_fns_a)

    def random_resized_crop(self, image_a, image_b):
        # random crop
        crop = transforms.RandomResizedCrop(self.final_size)
        params = crop.get_params(image_a, scale=self.resize_scale, ratio=self.resize_ratio)
        image_a = TF.crop(image_a, *params)
        image_b = TF.crop(image_b, *params)

        # resize
        resize = transforms.Resize(size=(self.final_size, self.final_size))
        image_a = resize(image_a)
        image_b = resize(image_b)
        return image_a, image_b

    def random_color_jitter(self, image_a, image_b):
        if random.random() > (1. - self.p_color_jitter):
            cj = transforms.ColorJitter(*self.color_jitter_param)
            color_jitter = cj.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)
            image_a = color_jitter(image_a)
            image_b = color_jitter(image_b)
        return image_a, image_b

    def random_grayscale(self, image_a, image_b):
        if random.random() > (1. - self.p_grayscale):
            gray_scale = transforms.Grayscale(num_output_channels=3)
            image_a = gray_scale(image_a)
            image_b = gray_scale(image_b)
        return image_a, image_b

    def random_gaussian_blur(self, image_a, image_b):
        if random.random() > (1. - self.p_gaussian_blur):
            gaussian_blur = GaussianBlur([.1, 2.])
            image_a = gaussian_blur(image_a)
            image_b = gaussian_blur(image_b)
        return image_a, image_b

    def random_horizontal_flip(self, image_a, image_b):
        if random.random() > (1. - self.p_horizontal_flip):
            image_a = TF.hflip(image_a)
            image_b = TF.hflip(image_b)
        return image_a, image_b

    def __getitem__(self, index):
        # load both images
        image_fn_a = self.image_fns_a[index]
        image_fn_b = self.image_fns_b[index]
        sample_a = self.loader(image_fn_a)
        sample_b = self.loader(image_fn_b)

        # apply same random operations on both
        sample_a, sample_b = self.random_resized_crop(sample_a, sample_b)
        sample_a, sample_b = self.random_color_jitter(sample_a, sample_b)
        sample_a, sample_b = self.random_grayscale(sample_a, sample_b)
        sample_a, sample_b = self.random_gaussian_blur(sample_a, sample_b)
        sample_a, sample_b = self.random_horizontal_flip(sample_a, sample_b)

        # convert to tensor and normalize
        sample_a = transforms.ToTensor()(sample_a)
        sample_b = transforms.ToTensor()(sample_b)
        sample_a = transforms.Normalize(mean=self.mean_a, std=self.std_a)(sample_a)
        sample_b = transforms.Normalize(mean=self.mean_b, std=self.std_b)(sample_b)
        return sample_a, sample_b


def find_mean_std(image_dir):
    import numpy as np
    import torch
    import torchvision.datasets as datasets

    dataset = datasets.ImageFolder(image_dir, transform=transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        num_workers=8,
        shuffle=False
    )
    data_mean = []  # Mean of the dataset
    data_std0 = []  # std of dataset
    data_std1 = []  # std with ddof = 1
    for data in loader:
        # shape (batch_size, 3, height, width)
        numpy_image = data[0].numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        data_mean.append(batch_mean)
        data_std0.append(batch_std0)
        data_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    data_mean = np.array(data_mean).mean(axis=0)
    data_std0 = np.array(data_std0).mean(axis=0)
    data_std1 = np.array(data_std1).mean(axis=0)
    return data_mean, data_std0


def main():
    # # mean1, std1 = find_mean_std(image_dir='/mnt/data_ssd/datasets/test')
    # mean1, std1 = find_mean_std(image_dir='/mnt/data_ssd/datasets/A2Face/customize_data_v9_55000_openface_aligned')
    # mean2, std2 = find_mean_std(image_dir='/mnt/data_ssd/datasets/A2Face/customize_data_v9_55000_config')
    # print(f'mean1: {mean1}, std1: {std1}')
    # print(f'mean2: {mean2}, std2: {std2}')

    import numpy as np
    import torch
    import time
    from PIL import Image

    def convert_image(image):
        img = image.numpy()
        img = np.clip(img * 255.0, 0.0, 255.0)
        img = np.transpose(img, axes=[1, 2, 0])
        img = Image.fromarray(np.uint8(img))
        return img

    # data_base_dir = '/mnt/data_ssd/datasets/A2Face/'
    data_base_dir = '/mnt/data_ssd/datasets/test'
    data_a = os.path.join(data_base_dir, 'a')
    data_b = os.path.join(data_base_dir, 'b')

    dataset = CustomDataset(data_a, data_b)
    # for a, b in dataset:
    #     img_a = convert_image(a)
    #     img_a.show()
    #     img_b = convert_image(b)
    #     img_b.show()
    #     time.sleep(1)
    #     print()

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=True)
    for i, (a, b) in enumerate(loader):
        img_a = convert_image(a[0])
        img_a.show()
        img_b = convert_image(b[0])
        img_b.show()
        time.sleep(1)
        print()
    return


if __name__ == '__main__':
    main()
