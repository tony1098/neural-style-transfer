# CSE 455 Final Project

## Problem Statement

In this project, we look at the problem of Neural Style Transfer. In the Neural Style Transfer problem, we have two input images, a content image and a style image. The goal is to take the style of the style image and apply it on our content image using Convolutional Neural Networks. 

We first take a look at the paper and play around with the implementation from Gatys et al., A Neural Algorithm of Artistic Style, 2015 which uses transfer learning by using pretrained VGG-19. We try exploring the impact of the parameters such as the weighting between style and content or how the style weighting on shallow or deep layers of the network may affect the results. We then compare and see how the results differ when when we use a different network than VGG to the transfer learning process.

## Dataset

## Background

## Method

Neural Style Transfer tries to minimize a loss function comprised of two terms, a content loss and a style loss. The content loss is rather straightforward - it's just the mean squared error of the feature map between the content image and generated image at a specific layer in the pretrained VGG-19 network (CAN PLAY AROUND WITH THIS!!!!). 

For the style of an image, we first compute Gram matrices using different feature maps in the VGG network. We concatenate the set of Gram matrices which then represent the style of an image. This is because the Gram matrix represent the covariance between different features/textures for different layers. We then take the concatenated Gram matrices for the content image and generated image and compute the mean squared error.

The following experiments are done based on the [reference implementation](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) of the paper from Gatys et al., A Neural Algorithm of Artistic Style, 2015. We take an image of the UW Suzzallo Library as the content image, the Van Gogh painting The Starry Night as the style image, and we get the generated image on the right after running 300 iterations of gradient descent.
![Image](images/suzzallo_starry_night.jpg)

We then try tweaking the weighting between the content and style losses. α represents the weighting of the content loss while β represents the weighting of the style loss. Previously, we set α/β = 10^-6. As we can see in the experiment below, when α/β is small, most of the content image is preserved, while with large α/β, we loose more details of the structure in the content image.
![Image](images/alpha_beta_comparison.jpg)

Next, we change the the layers in which the style loss function is computed for. Previously, we used the first five convolutional layers of the VGG-19 network in the style loss. We compare that to using on the first convolutional layer and the fifth convolution layer in the style loss function.


### Results

### Code

See [Google Colab notebook](PUT LINK!!!!!!!!!!!!)

## Summary

## References

[Basic Neural Style Transfer implementation](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

Gatys et al., A Neural Algorithm of Artistic Style, 2015


```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/tony1098/tony1098.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
