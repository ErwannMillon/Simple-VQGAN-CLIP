# Simple VQGAN CLIP
This is a very simple VQGAN-CLIP implementation that I built as a part of my <a href= "https://github.com/ErwannMillon/face-editor"> Face Editor project </a> . This simplified version allows you to generate or edit images using text with just three lines of code. For a more full featured implementation with masking, more advanced losses, and a full GUI, check out the Face Editor project. 

By default this uses a CelebA checkpoint (for generating/editing faces), but also has an imagenet checkpoint that can be loaded by specifying vqgan_config and vqgan_checkpoint when instantiating VQGAN_CLIP. 

Learning rate and iterations can be set by modifying vqgan_clip.lr and vqgan_clip.iterations . 

You can edit images by passing `image_path` to the generate function. 
See the generate function's docstring to learn more about how to format prompts. 

## Usage
The easiest way to test this out is by <a href="https://colab.research.google.com/drive/1Ez4D1J6-hVkmlXeR5jBPWYyu6CLA9Yor?usp=sharing
">using the Colab demo</a>

To install locally: 
- Clone this repo
- Install git-lfs (ubuntu: sudo apt-get install git-lfs , MacOS: brew install git-lfs) 

In the root of the repo run:

```
conda create -n vqganclip python=3.8
conda activate vqganclip
git lfs install
git submodule init
git submodule update --init --recursive
pip install -r requirements.txt
cd model_checkpoints && git pull origin main
```

### Generate new images
```
from VQGAN_CLIP import VQGAN_CLIP
vqgan_clip = VQGAN_CLIP()
vqgan_clip.generate("a picture of a smiling woman")
```

### Edit an image
```
from VQGAN_CLIP import VQGAN_CLIP
vqgan_clip = VQGAN_CLIP()
vqgan_clip.generate("a picture of a smiling woman",
                    image_path="sad_woman.jpg")
```

### Make an animation from the most recent generation
`vqgan_clip.make_animation()`

## Features:
- Positive and negative prompts
- Multiple prompts
- Prompt Weights
- Creating GIF animations of the transformations
- Wandb logging



