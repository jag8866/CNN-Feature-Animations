from fastai.conv_learner import *
import heapq

class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()

#layer = 40
#filter = 64  # will be marked with a vertical line in the plot
#total_filters_in_layer = 512

#model - the model to use
#image - the image to find the activations for
#exclude - layer(s) and filter(s) to ignore in format [(30, 145), (38, 20), ...]
def getLargestActivations(model, image, exclude, sz, layers = [28, 30, 33, 35, 38, 40]):


    #model = vgg16(pre=True).cuda().eval()
    #set_trainable(model, False)

    #sz = 224
    train_tfms, val_tfms = tfms_from_model(vgg16, sz)

    transformed = val_tfms(np.array(image)/255)

    largest_activations_means = []

    for layer in layers:
        activations = SaveFeatures(list(model.children())[layer])

        model(V(transformed)[None]);

        #mean_act = [activations.features[0,i].mean().item() for i in range(total_filters_in_layer)]
        largest_activations_means += heapq.nlargest(5, [( (layer, i), activations.features[0, i].mean().item() ) for i in range(len(activations.features[0])) if (layer, i) not in exclude], key=lambda x: x[1])

        activations.close()

    return [x[0] for x in heapq.nlargest(5, largest_activations_means, key=lambda x: x[1])]
