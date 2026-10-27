import yaml
import math
import random
import argparse

from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from huggingface_hub import HfApi, snapshot_download
from ultralytics import YOLO
from omegaconf import OmegaConf


# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():

    parser = argparse.ArgumentParser(
        description='Train YOLO model for obstacle detection'
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', dest='train', action='store_true')
    mode.add_argument('--test', dest='test', action='store_true')
    mode.add_argument('--show', type=int)

    return parser.parse_args()


def show_images(images):
    n = len(images)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, img in zip(axes.ravel(), images):
        ax.imshow(img)

    plt.tight_layout()
    plt.show()



def main():
    # set_seed(42)
    args = parse_args()
    with open("config.yaml", 'r') as file:
        config = OmegaConf.create(yaml.safe_load(file))

    api = HfApi()

    # git clone https://huggingface.co/datasets/Abtinzandi/Obstacle-Detection-Dataset-YOLO

    ds = config.dataset_path

    print(f"Датасет в: {ds}")

    if args.show:
        number = args.show
        images_list = list((Path(ds) / "train" / "images").rglob("*.jpg"))

        ids = random.sample(range(0, len(images_list)), min(number, len(images_list)))
        images = [Image.open(images_list[i]) for i in ids]
        show_images(images)
    elif args.test:
        number = 4
        path = config.weights_path
        model = YOLO(path)
    else:
        model = YOLO(config.model)
        train(model, ds, config)

        images_list = list((Path(ds) / "train" / "images").rglob("*.jpg"))

        ids = random.sample(range(0, len(images_list)), min(number, len(images_list)))
        images = [Image.open(images_list[i]) for i in ids]
        images = [model(img)[0].plot()[:, :, ::-1] for img in images]
        show_images(images)


def train(model, ds, config):
    model.train(data=str(Path(ds) / "data.yaml"), epochs=config.epochs)


if __name__ == "__main__":
    main()