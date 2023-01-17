# Face Editor

## Examples


## Overview
This interactive GUI face editor uses a CelebA-pretrained VQGAN-CLIP for prompt-based image manipulation, as well as slider based manipulation using extracted latent vectors. I built this in a few days as part of the Ben's Bites December 22 Hackathon

I've since written a series of Medium articles which provide a detailed and beginner-friendly explanation of how this was built and the intuition behind latent image manipulation. 

[Coming Soon]

## Demo
Clone the repo and run `app.py` or <a href="https://colab.research.google.com/drive/110uAZIRQjQen0rKqcnX_bqUXIahvRsm9?usp=sharing"> open this colab notebook </a> and run all the cells, then click on the link that appears under the final cell. 

## Features:
- Positive and negative prompts
- Multiple prompts
- Local editing using a gradient masked adversarial loss (implemented as custom pytorch backpropagation hooks). The CLIP loss gradients are masked according to the user's selection, and the LPIPS loss gradients are masked with the inverse of the user mask in order to preserve the initial identity of the face and prevent changes outside of the masked zone that otherwise happen due to latent variable entanglement. 
- Extracted latent vectors for slider-based editing
- Rewinding through the history of edits, resuming edits from a previous point in the history
- Creating GIF animations of the editing process 

## Future work / ideas
- Implementing an LRU cache to the render_all_transformations function. I first implemented a simple cache that refactored the function to take the transformations as arguments and cached the corresponding decoded transformed image to speed up rewinds through the prompt editing history, but this was very memory inefficient. An LRU cache could mitigate this, or even a cache that only caches the most recent prompt edit. Might add this later. 

