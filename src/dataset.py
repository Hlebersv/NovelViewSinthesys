import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional
import PIL
from PIL import Image
import math


class FFHQDataset(Dataset):
    def __init__(
            self,
            path_to_img,
            dataframe,
            size=512,
            interpolation="bicubic",
            feature_name='name',
    ):

        self.feature_name = feature_name

        self.dataframe_pose = dataframe
        self.size = size

        self.names = self.dataframe_pose[feature_name].values

        self.file_names = self.dataframe_pose[feature_name].apply(lambda file_name: f'{path_to_img}/{file_name}.png').values

        self.num_images = len(self.file_names)
        self._length = self.num_images

        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

    def __len__(self):
        return self._length

    def read_pose(self, name):

        p_x, p_y = self.dataframe_pose[self.dataframe_pose[self.feature_name] == name].values[0][:2]
        p = torch.Tensor([p_x,p_y]).unsqueeze(0)
        p_flip = torch.Tensor([p_x, -p_y + math.pi]).unsqueeze(0)

        return p.float(),p_flip.float()

    def __getitem__(self, i):
        example = {}

        image = Image.open(self.file_names[i])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        example["pose"], example["pose_flip"] = self.read_pose(self.names[i])
        example['diff_pose'] = example["pose_flip"] - example["pose"]

        tformvae = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (self.size, self.size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711])  # magic constants
        ])

        example["pixel_values"] = tformvae(image)
        example["pix_for_vit"] = functional.hflip(tformvae(image))

        flip = (torch.rand(1) < 0.5)

        if flip:
            example["pixel_values"], example["pix_for_vit"] = example["pix_for_vit"], example["pixel_values"]
            example['diff_pose'] = - example['diff_pose']

        example["pix_for_vit"] = transforms.Resize((224, 224))(example["pix_for_vit"])

        return example
