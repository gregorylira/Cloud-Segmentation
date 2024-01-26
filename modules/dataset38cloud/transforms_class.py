import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt

class DataAugmentations():
    def get_train_transforms(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ], p=1.0)

    def get_valid_transforms(self):
        return A.Compose([
        ], p=1.0)

    def get_test_transforms(self):
        return A.Compose([
        ], p=1.0)
    
    def visualize(self,image, mask, original_image=None, original_mask=None,name_fig = ""):
        fontsize = 18
        
        if original_image is None and original_mask is None:
            f, ax = plt.subplots(2, 1, figsize=(8, 8))

            ax[0].imshow(image)
            ax[1].imshow(mask)
            f.savefig(f'{name_fig}_batch.png')
        else:
            f, ax = plt.subplots(2, 2, figsize=(8, 8))

            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title('Original image', fontsize=fontsize)
            
            ax[1, 0].imshow(original_mask)
            ax[1, 0].set_title('Original mask', fontsize=fontsize)
            
            ax[0, 1].imshow(image)
            ax[0, 1].set_title('Transformed image', fontsize=fontsize)
            
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
            f.savefig(f'{name_fig}_batch.png')