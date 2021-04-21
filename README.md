# XDVioDet
Official implementation of "**Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision**" ECCV2020.

The project website is [XD-Violence](https://roc-ng.github.io/XD-Violence/). The features can be downloaded from our project website.

where we oversample each video frame with the “5-crop” augment, “5-crop” means cropping images into the center and four corners. _0.npy is the center, _1~ _4.npy is the corners.

## How to train
  * download or extract the features.
  * use make_list.py in the list folder to generate the training and test list.
  * change the parameters in option.py
  * run main.py
  
## How to test
  * run infer.py
  
  &nbsp; &nbsp; &nbsp;  *the model is in the ckpt folder.*

Thanks for your attention!
