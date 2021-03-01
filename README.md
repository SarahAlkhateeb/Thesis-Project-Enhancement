# Thesis project "Ocean Exploration with Artificial Intelligence" 

This repository contains the code used in our thesis project to enhance the data. This repo is a copy from the [offical pytorch implementation of Cyclegan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). 



#### Training 
Do: 

- `conda env create -f environment.yml` 
- `conda activate pytorch-CycleGAN-and-pix2pix`
- Create dataset folder `coral2coral` under `/dataset` with subfolders `trainA`, and `trainB`. Place bad quality images into `trainA` and good quality images into `trainB`.
- train for 100 epochs (default 200):`python train.py --dataroot ./datasets/coral2coral --name c2c_cyclegan --model cycle_gan --n_epochs 50 --n_epochs_decay 50` 
- continue training from checkpoints after training of 100 epochs: `python train.py -- --dataroot ./datasets/coral2coral --name c2c_cyclegan --model cycle_gan --continue_train --epoch_count 101` 

#### Testing
Do:
- Create subfolder datasets/coral2coral/testA and place test images into the folder. 
- Go to checkpoints and remove "A" from last generator checkpoint naming (to be able to run 1 side test)
- `python test.py -- --dataroot datasets/coral2coral/testA --name c2c_cyclegan --model test --no_dropout`


**Note**: cyclegan scales and crops images to size 256x256. 
 change sizes with `--load_size` and `--crop_size`



