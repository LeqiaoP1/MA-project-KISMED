# Set-up on Ubuntu 22.04.5 LTS

## Dependencies

This codebase has been tested with the packages and versions specified in `requirements.txt` and Python 3.11.

Creating a new "virtual environment" via python (not "conda" just as the HPC Cluster suggests):

```bash
python -m venv .venv

source .venv/bin/activate
```

Upgrade pip and build tools

```Shell
python -m pip install --upgrade \
    pip \
    setuptools \
    wheel
```

Then, install [PyTorch](https://pytorch.org/) (2.11.0+cu128) and torchaudio torchvision (0.26.0+cu128) which are built with targeted cuda 12.8.

```bash
pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

Verify the installed PyTorch 2.11 and CUDA version 12.8.

```Shell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Install common ML packages

```Shell
pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    pillow \
    opencv-python \
    scikit-learn \
    tqdm \
    pyyaml \
    jupyter \
    ipykernel \
    tensorboard \
    timm \
    wandb \
    einops \
    fvcore \
    iopath \
    albumentations \
    h5py
```

### Install the Detectron2 from the source

as Detectron2 needs a C++ compiler to build. Some tools must be installed directly on the system

```bash
# Update Ubuntu's package manager and install the essential C++ compiler (gcc/g++)
sudo apt-get update && sudo apt-get install -y build-essential libsm6 libglib2.0-0 libxrender1 libxext6 ninja-build
```

Following apply inside the active env

```bash
# 1. Install prerequisites required by Meta AI
pip install cython pycocotools
pip install 'git+https://github.com/facebookresearch/fvcore'

# 2. Compile Detectron2 directly using your active environment's PyTorch links
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

ℹ️ If data loading and image transforms are the bottleneck, consider replacing Pillow with [Pillow-SIMD](https://github.com/uploadcare/pillow-simd) and compiling it with [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo). You can find a detailed guide on how to do this [here](https://fastai1.fast.ai/performance.html#installation) or use the provided script:

```bash
sh tools/install_pillow_simd.sh
```

## Dataset Preparation

### Dataset structure

For simplicity and uniformity, all our datasets are structured in the following way:

```
/path/to/data/
├── train/
│   ├── modality1/
│   │   └── subfolder1/
│   │       ├── img1.ext1
│   │       └── img2.ext1
│   └── modality2/
│       └── subfolder1/
│           ├── img1.ext2
│           └── img2.ext2
└── val/
    ├── modality1/
    │   └── subfolder2/
    │       ├── img3.ext1
    │       └── img4.ext1
    └── modality2/
        └── subfolder2/
            ├── img3.ext2
            └── img4.ext2
```

The folder structure and filenames should match across modalities.
If a dataset does not have specific subfolders, a generic subfolder name can be used instead (e.g., `all/`).

For most experiments, we use RGB  (`rgb`), depth (`depth`), and semantic segmentation (`semseg`) as our modalities.

RGB images are stored as either PNG or JPEG images.
Depth maps are stored as either single-channel JPX or single-channel PNG images.
Semantic segmentation maps are stored as single-channel PNG images.

### Datasets

We use the following datasets in our experiments:

- [**ImageNet-1K**](https://www.image-net.org/)
- [**ADE20K**](http://sceneparsing.csail.mit.edu/)
- [**Hypersim**](https://github.com/apple/ml-hypersim)
- [**NYUv2**](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [**Taskonomy**](https://github.com/StanfordVL/taskonomy/tree/master/data)

To download these datasets, please follow the instructions on their respective pages.
To prepare the NYUv2 dataset, we recommend using the provided [`prepare_nyuv2.py`](tools/prepare_nyuv2.py) script.

### Downloadable ImageNet-1K pseudo labels

We publish links to download the Omnidata depth and COCO semantic segmentation pseudo labels [here](https://github.com/EPFL-VILAB/MultiMAE/tree/main/tools/pseudolabel_links).
The images for each ImageNet class are stored as tar-files.

To download the dataset, we recommend using aria2c, which you can install using:

```
sudo apt-get update
sudo apt-get install aria2
```

Download both train and validation splits for the depth and semantic segmentation labels by calling

```
aria2c --input-file ./tools/pseudolabel_links/all_aria2c.txt -d /the/download/directory -j 16 -x 16
```

For additional download options, please see the [aria2c documentation](http://aria2.github.io/manual/en/html/aria2c.html).

Please note that by downloading this dataset you are consenting to non-commercial use and the license.

### Pseudo labeling networks

ℹ️ The MultiMAE pre-training strategy is flexible and can benefit from higher quality pseudo labels and ground truth data. So feel free to use different pseudo labeling networks and datasets than the ones we used!

We use two off-the-shelf networks to pseudo label the ImageNet-1K dataset.

- **Depth estimation**: We use a [DPT](https://arxiv.org/abs/2103.13413) with a ViT-B-Hybrid backbone pre-trained on the [Omnidata](https://omnidata.vision/) dataset. You can find installation instructions and pre-trained weights for this model [**here**](https://docs.omnidata.vision/pretrained.html).
- **Semantic segmentation**: We use a [Mask2Former](https://bowenc0221.github.io/mask2former/) with a Swin-S backbone pre-trained on the [COCO](https://cocodataset.org/) dataset. You can find installation instructions and pre-trained weights for this model [**here**](https://github.com/facebookresearch/Mask2Former).

For an example of how to use these networks for pseudo labeling, please take a look at our [**Colab notebook**](https://colab.research.google.com/github/EPFL-VILAB/MultiMAE/blob/main/MultiMAE_Demo.ipynb).
