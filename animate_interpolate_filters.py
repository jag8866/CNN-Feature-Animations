from fastai.conv_learner import *
from cv2 import resize
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import math
from scipy import signal
import time
from datetime import datetime
import cv2
import random as r
from getLargestActivations import getLargestActivations
import numpy as np

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True).cuda()

    def close(self):
        self.hook.remove()

class AnimatedGif:
    def __init__(self, size=(640, 480)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []

    def add(self, image, label=''):
        plt_im = plt.imshow(image, cmap='Greys', vmin=0, vmax=1, animated=True)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])

    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=22)

#Takes a list of layers/filters in the standard format - [(30, 141), (38, 12), ...] and returns a random
#combination of a selection of them in the format accepted by visualize() - [(30, 141), (38, 12)] or [(30, 141, .5), (38, 12, .5)]
#activations must be at least 5 items long
def randomCombination(activations):
    combo = random.choice([
        [activations.pop(random.randrange(1, len(activations))) for i in range(random.randrange(1, 4))],
        [activations.pop(random.randrange(1, len(activations))) + (random.randrange(100)/100,), activations.pop(random.randrange(0, len(activations)))]
    ])
    if len(combo[0]) == 3:
        combo[1] = combo[1] + (1 - combo[0][2],)
    return combo

class FilterVisualizer():
    def __init__(self, size=56, upscaling_steps=12, upscaling_factor=1.2, tile=False):
        self.size, self.upscaling_steps, self.upscaling_factor = size, upscaling_steps, upscaling_factor
        #Load the model
        self.model = vgg16(pre=True).cuda().eval()
        #Does not need to be trained (we are just using the pretrained weights)
        set_trainable(self.model, False)
        #Generate the random noise starting image - needs to be bigger if we are create tileable textures
        if tile:
            self.start_image = np.uint8(np.random.uniform(120, 180, (self.size * 2, self.size * 2, 3))) / 255  # generate random image
        else:
            self.start_image = np.uint8(np.random.uniform(120, 180, (self.size, self.size, 3))) / 255  # generate random image

    def optimize_image(self, img, opt_steps, activaction_dict, filter1, filter2, w1, w2, step=0, lr=0.1):
        img_var = V(self.val_tfms(img)[None], requires_grad=True)  # convert image to Variable that requires grad
        self.optimizer = torch.optim.Adam([img_var], lr=lr, weight_decay=1e-6)

        # Here we optimize the pixel values with gradient descent in attempt to maximize our current set of filters
        for n in range(opt_steps):
            self.optimizer.zero_grad()
            self.model(img_var)

            # Smooth filter mixing with weights like [(40, 265, .5), ...]
            if len(filter1[0]) == 3:
                loss1 = sum([-activaction_dict[filt[0]].features[0, filt[1]].mean() * filt[2] for filt in filter1])
            # Layered filter mixing like [(40, 265), ...]
            else:
                # Current filter to use
                filt = int((step / (self.upscaling_steps)) * len(filter1))
                loss1 = -activaction_dict[filter1[filt][0]].features[0, filter1[filt][1]].mean()

            # Smooth filter mixing
            if len(filter2[0]) == 3:
                loss2 = sum([-activaction_dict[filt[0]].features[0, filt[1]].mean() * filt[2] for filt in filter2])
                # Layered filter mixing (low to high)
            else:
                # Current filter to use, steps to the right as we move upwards in base
                filt = int((step / (self.upscaling_steps)) * len(filter2))
                loss2 = -activaction_dict[filter2[filt][0]].features[0, filter2[filt][1]].mean()

            # Our loss value combines filter1 and filter2 to allow smooth animation between them by sliding weights
            loss = (w1 * loss1) + (w2 * loss2)

            loss.backward()
            self.optimizer.step()

        img = self.val_tfms.denorm(img_var.data.cpu().numpy()[0].transpose(1, 2, 0))

        return img

    #Creates a filter visualization. Final output is stored in self.output, returns buildframes which has all the smaller
    #images used to create the final one
    #filter1 and filter2 will be blended according to the corresponding weights w1 and w2
    #Filters are either fully mixed using triples with weights as the last value or mixed by base (lower bases will use
    #earlier filters).
    #Example of type 1: filter1 = [(40, 69, .5), (34, 420, .5)]
    #Example of type 2: filter2 = [(40, 69), (34, 420)]
    #(First value in tuple is always layer, second is always filter (the number of the particular neuron))
    def visualize(self, activaction_dict, prev_frame, filter1, filter2, start_image, w1=.5, w2=.5, lr=0.1, opt_steps=30, blur=None, base=0, tile=False, splitscaling=False):
        #start_image should be passed as self.start_image unless this is not the first frame of an animation
        img = start_image

        #Create an empty list where we will place the smaller images used to make our final image
        buildframes = [None] * self.upscaling_steps

        if prev_frame is not None:
            #Blend current image with final image of previous frame (makes smoother animation)
            img = (.7 * img) + cv2.resize((.3 * prev_frame), (img.shape[0], img.shape[0]), interpolation=cv2.INTER_CUBIC)\

        sz = self.size
        if base != 0:
            sz = img.shape[0]

        #For tiling we must create a tiled version of our image and optimize that for each step, here is the initial one
        if tile and prev_frame is not None:
            #Add copies of the image on to each side and diagonal of itself
            horiz = np.concatenate([img, img, img], 1)
            img = np.concatenate([horiz, horiz, horiz], 0)

            #Now split out a portion of this, so we have the original image with a portion of its copies around it
            #Much faster than if we tried to optimize the 9x size one
            horiz = np.split(img, 6, 0)
            scronch = np.concatenate([horiz[1], horiz[2], horiz[3], horiz[4]], 0)
            vert = np.split(scronch, 6, 1)
            img = np.concatenate([vert[1], vert[2], vert[3], vert[4]], 1)

        #We start with a very small image, each step we scale the image up and optimize it again
        #This helps have a variety of structure sizes and more detail
        #We can make smoother and slower animation by having base higher than 0 and using the buildframes of the
        #previous frame
        for step in range(base, self.upscaling_steps):  # scale the image up upscaling_steps time
            #Use lower layers for later steps to refine small details
            #if step >= self.upscaling_steps - 1:
            #    smallfilts = getLargestActivations(self.model, img, filter1 + filter2, sz, layers=[14])
            #    filter1, filter2 = [smallfilts[0]], [smallfilts[1]]

            if tile:
                #Must adjust size if tiling
                train_tfms, self.val_tfms = tfms_from_model(vgg16, sz * 2)
            else:
                train_tfms, self.val_tfms = tfms_from_model(vgg16, sz)

            #This is my sad attempt at making the animation a bit cleaner by making later steps (smaller details on image)
            #move less by having less optimization steps
            if step == self.upscaling_steps - 2:
               opt_steps = 15
               blur = blur - 2
            if step >= self.upscaling_steps - 1:
               opt_steps = 1
               blur = blur - 1



            img = self.optimize_image(img, opt_steps, activaction_dict, filter1, filter2, w1, w2, step=step, lr=lr)



            #Split the tiled image back to its original size for now
            if tile:
                horiz = np.split(img, 4, 0)
                scronch = np.concatenate([horiz[1], horiz[2]], 0)
                vert = np.split(scronch, 4, 1)
                img = np.concatenate([vert[1], vert[2]], 1)

            #New size of next step's image
            sz = int(self.upscaling_factor * sz)

            #(For tiling) round sz to a value divisible by 2
            if tile:
                sz = int(2 * math.ceil(sz / 2))

            self.output = img

            buildframes[step] = img

            #Scale up the image to its next size
            img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)  # scale image up

            #Blend the current image with the previous frame for smoother animation
            if prev_frame is not None:
                if step % 2 == 0:
                    img = (.7 * img) + cv2.resize((.3 * prev_frame), (img.shape[0], img.shape[0]), interpolation=cv2.INTER_CUBIC)
                else:
                    img = (.95 * img) + cv2.resize((.05 * prev_frame), (img.shape[0], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            #Repeat the same tiling process from the beginning
            if tile:
                horiz = np.concatenate([img, img, img], 1)
                img = np.concatenate([horiz, horiz, horiz], 0)

                horiz = np.split(img, 6, 0)
                scronch = np.concatenate([horiz[1], horiz[2], horiz[3], horiz[4]], 0)
                vert = np.split(scronch, 6, 1)
                img = np.concatenate([vert[1], vert[2], vert[3], vert[4]], 1)

            #Apply a slight blur - helps to reduce high frequency noise
            if blur is not None:
                img = cv2.blur(img, (blur, blur))


        if splitscaling:
            sc_sz = int( (self.upscaling_factor * sz) * 3 )
            cv2.resize(img, (sz, sz), interpolation=cv2.INTER_CUBIC)

            splonk = np.split(img, 6, 0)
            split = [np.split(splonk[i], 6, 1) for i in splonk]

            for x in range(6-2):
                for y in range(6-2):

                    sliding_window = np.concatenate([
                        np.concatenate([split[x][y], split[x+1][y], split[x+2][y]], 0),
                        np.concatenate([split[x][y + 1], split[x + 1][y + 1], split[x + 2][y + 1]], 0),
                        np.concatenate([split[x][y + 2], split[x + 1][y + 2], split[x + 2][y + 2]], 0)
                        ])

                    self.optimize_image(img_var, opt_steps, activaction_dict, filter1, filter2, w1, w2, step=step+1)

            buildframes[step+1] = img


        return buildframes

    def save(self):
        plt.imsave(datetime.now().strftime("%m%d%Y%H%M%S") + ".jpg", np.clip(self.output, 0, 1))

#If selfchanging is true, first item in filters will start and each after will be chosen from the most activated features from the last
def animate(filters=[[(35, 0)]], selfchanging=False, zoomrate = 0, tile=False, splitscaling=False):

    start_filter = filters[0]

    #FV = FilterVisualizer(size=56, upscaling_steps=12, upscaling_factor=1.2, tile=True)
    if tile:
        FV = FilterVisualizer(size=30, upscaling_steps=15, upscaling_factor=1.2, tile=tile)
    else:
        FV = FilterVisualizer(size=56, upscaling_steps=12, upscaling_factor=1.2, tile=tile)
    #gif = AnimatedGif(size=(408,408))

    if tile:
        gif = AnimatedGif(size=(408*3,408*3))
    else:
        gif = AnimatedGif(size=(408, 408))

    activaction_dict = {}

    for filter in [28, 30, 33, 35, 38, 40]:
        activaction_dict[filter] = SaveFeatures(list(FV.model.children())[filter])

    buildframes = FV.visualize(activaction_dict, None, start_filter, start_filter, FV.start_image, base=0, w1=1, w2=0, blur=5, tile=tile, splitscaling=splitscaling)

    if selfchanging:
        filter1, filter2 = start_filter, randomCombination(getLargestActivations(FV.model, buildframes[FV.upscaling_steps - 1], start_filter, 400))

    divisions = 35
    totalframes = 8000
    try:
        for i in range(totalframes):
            base = 0

			#We must pass the previous frame into visualize() so it can be used to generate the next frame such that they are similar enough to animate smoothly
            previous_frame = buildframes[FV.upscaling_steps - 1]
			
			#Handle zooming by scaling up the last frame
            if zoomrate != 0:
                previous_frame = cv2.resize(previous_frame[0 + zoomrate:previous_frame.shape[0] - zoomrate, 0 + zoomrate:previous_frame.shape[0] - zoomrate], (previous_frame.shape[0], previous_frame.shape[0]), interpolation=cv2.INTER_CUBIC)

			#If selfchanging is true we select randomized sets of filters to blend between
            if selfchanging and i%divisions == 0 and i != 0:
                filter1, filter2 = filter2, randomCombination(getLargestActivations(FV.model, previous_frame, filter1 + filter2, 400))

			#Otherwise we blend between the filters given
            if not selfchanging:
                filter1, filter2 = filters[math.floor(i / divisions)], filters[math.floor((i) / divisions) + 1]

			#Call visualize to create a frame. Note the weight arguments, w1 and w2, we switch the weight back and forth between them as the animation plays
            newbuildframes = FV.visualize(activaction_dict, previous_frame, filter1, filter2, buildframes[base], base=base, w1=1 - (i % divisions) / divisions, w2=(i % divisions) / divisions, blur=5, tile=tile, opt_steps=30, splitscaling=splitscaling)


            for j in range(len(newbuildframes)):
                if newbuildframes[j] is not None:
                    buildframes[j] = newbuildframes[j]

            # current_frame = np.uint8(np.clip((((FV.output) * (255.0) / (FV.output.max()) + 70) * 255 / 325), 0, 255))
            current_frame = FV.output


			#Handle tiling by making copies of the image on each side.
            if tile:
                horiz = np.concatenate([current_frame, current_frame, current_frame], 1)
                x = np.concatenate([horiz, horiz, horiz], 0)
                gif.add(x)  # , label=str(filter1) + ", " + str(filter2))
            else:
                x = current_frame
                gif.add(x)

            #Add frame to our gif
            if i == totalframes - 1:
                gif.save(datetime.now().strftime("%m%d%Y%H%M%S") + ".gif")
                break
            elif (i + 1) % 100 == 0:
                gif.save(datetime.now().strftime("%m%d%Y%H%M%S") + "_progress" + str(math.floor(i / 100)) + ".gif")

        for activation in activaction_dict.values():
            activation.close()
    except:
        gif.save(datetime.now().strftime("%m%d%Y%H%M%S") + ".gif")

if __name__ == "__main__":
	#This will create an animation moving through the sets of filters below in order
    filters = [[(35, 96)], [(35, 102)], [(35, 366)], [(35, 447)], [(35, 426)], [(35, 374)], [(35, 460)], [(35, 481)], [(35, 481)], [(35, 89)], [(35, 83)], [(35, 86)], [(35, 82)], [(30, 374)], [(30, 367)], [(30, 343)], [(30, 320)], [(30, 328)], [(30, 281)], [(30, 236)], [(30, 213)], [(30, 141)], [(30, 128)]]
    animate(filters, zoomrate=4, tile=False, splitscaling=False)
