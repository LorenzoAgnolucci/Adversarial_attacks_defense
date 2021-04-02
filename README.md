# Adversarial attacks defense


## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [Acknowledgments](#acknowledgments)

## About The Project
Based on the paper "Neural Compression Restoration Against Gradient-based Adversarial Attacks".

In this work the proposed defense strategy is evaluated against black-box attacks. In particular, we consider the [Hop Skip Jump](https://arxiv.org/abs/1904.02144) and the [Square](https://arxiv.org/abs/1912.00049) attack.

More details about the project in the [paper](paper.pdf) or in the [presentation](presentation.pdf).

### Built With

* [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
* [PyTorch](https://pytorch.org/)

## Installation

To get a local copy up and running follow these simple steps:

1. Clone the repo
```sh
git clone https://github.com/LorenzoAgnolucci/Adversarial_attacks_defense.git
```
2. Run ```pip install -r requirements.txt``` in the root folder of the repo to install the requirements

3. Run ```pip install -e adversarial-robustness-toolbox/``` in the root folder to install the ART module with the custom files


## Usage

1. Download the [dataset](http://www.image-net.org/download)

2. Change the path of the images and the parameters in ```jpeg_gan_hop_skip_jump_pytorch.py``` and ```jpeg_gan_square_pytorch.py```

3. Run ```jpeg_gan_hop_skip_jump_pytorch.py``` or ```jpeg_gan_square_pytorch.py``` to evaluate the defense strategy against the corresponding attack

## Authors

* [**Lorenzo Agnolucci**](https://github.com/LorenzoAgnolucci)

## Acknowledgments
Visual and Multimedia Recognition © Course held by Professor [Alberto Del Bimbo](https://scholar.google.it/citations?user=bf2ZrFcAAAAJ&hl=it) - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html)
