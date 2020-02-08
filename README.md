# CNN Feature Animations

Using a Convolutional Neural Network already trained to recognize a wide variety images, it is possible to generate an 
image that is optimized to maximize the response of one particular neuron in the network. This creates an image that is 
essentially a representation of what pattern/shape that particular neuron is trained to detect. I saw the beautiful images 
this method creates and decided to take it to the next level and create animations with them.

This project was based on the code from this tutorial:
https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030

I added the ability to create animations, blend smoothly between multiple filters, maximize multiple filters at once in two different
ways (maximizing the sums of their outputs or maximizing some at earlier stages of image generation and others at later ones), 
create zooming animations, and create tiled images and animations. 
