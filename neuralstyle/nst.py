#Import
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from torch import optim
from torchvision.utils import save_image
from PIL import Image
import os

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # The first number x in convx_y gets added by 1 after it has gone
        # through a maxpool, and the second y if we have several conv layers
        # in between a max pool. These strings (0, 5, 10, ..) then correspond
        # to conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 mentioned in NST paper
        self.chosen_features = ["0","5","10","19","28"]
        # We don't need to run anything further than conv5_1 (the 28th module in vgg)
        # Since remember, we dont actually care about the output of VGG: the only thing
        # that is modified is the generated image (i.e, the input).
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self,x):
        features = []
        # Go through each layer in model, if the layer is in the chosen_features,
        # store it in features. At the end we'll just return all the activations
        # for the specific layers we have in chosen_features

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features :
                features.append(x)

        return  features

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
imsize = 256
# Here we may want to use the Normalization constants used in the original
# VGG network (to get similar values net was originally trained on), but
# I found it didn't matter too much so I didn't end of using it. If you
# use it make sure to normalize back so the images don't look weird.
loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

original_img = load_image("annahathaway.png")
# style_img = load_image("style.jpg")

dir = 'styles'
styles = []
for files in os.listdir('styles'):
    print(os.path.join(dir,files))
    img = load_image(os.path.join(dir,files))
    styles.append(img)

original_img = load_image('annahathaway.png')

# initialized generated as white noise or clone of original image.
# Clone seemed to work better for me.

generated = original_img.clone().requires_grad_(True)

# Initialize model
model = VGG().to(device).eval()

# Hyperparameter
total_steps = 200
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated],lr=learning_rate)

# Train network
for idx,style_img in enumerate(styles):
    for step in range(total_steps):
        # Obtain the convolution features in specifically chosen layers
        generated_features = model(generated)
        original_features = model(original_img)
        style_features = model(style_img)

        #Loss
        style_loss = original_loss = 0
        for gen_feature,orig_feature,style_feature in zip(generated_features,original_features,style_features):

            original_loss += torch.mean((gen_feature - orig_feature) ** 2)
            batch_size,channel,height,width = gen_feature.shape
            G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())

            A = style_feature.view(channel,height * width).mm(
                style_feature.view(channel,height * width).t()
            )
            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * original_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(total_loss)
            save_image(generated,f"styles/generated_{idx+1}.png")

