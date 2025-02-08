import torch
from PIL import Image
import os
from torchvision import transforms

# TODO: Add PnP src


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed)
    )  # return a tuple

    image_grid = make_grid(images, rows=config.train_batch_size // 2, cols=config.train_batch_size // 2)
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def save_images(config, im, epoch):
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    im[0].save(f"{test_dir}/{epoch:04d}.png")


def prepare_inference_im(path):

    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711])])

    images = []
    for im in os.listdir(path):
        images.append(tform(Image.open(f"{path}/{im}")).unsqueeze(0))

    return images


