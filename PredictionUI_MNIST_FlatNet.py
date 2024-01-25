import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import gradio as gr

# DEFINE MODEL (required for loading weights)
class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

class FlatNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = FlattenLayer()
        self.linear1 = nn.Linear(1*28*28, 512)
        self.linear2 = nn.Linear(512, 10)

        self.activation_fn = nn.ReLU()

    def forward(self, x):
        x_flat = self.flatten(x)
        x_linear1 = self.linear1(x_flat)
        x_linear1_act = self.activation_fn(x_linear1)
        class_logits = self.linear2(x_linear1_act)
        return class_logits


# LOAD PRE-TRAINED MODEL
model = FlatNet()

checkpoint = torch.load('flatnet_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
# SWITCH MODEL TO PREDICTION ONLY MODE
model.eval()

# Same transforms that used in training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function for processing input image
# Since we're only interested in prediction, we disable the gradient computations
@torch.no_grad()
def recognize_digit(image):
    #print(type(image))
    #print(image.shape)
    image_tensor = transform(image) # 1, 28, 28
    image_tensor = image_tensor.unsqueeze(0) # add dummy batch dimension 1, 1, 28, 28
    #print(image_tensor.shape)
    
    logits = model(image_tensor)
    preds = F.softmax(logits, dim=1) # convert to probabilities
    preds_list = preds.tolist()[0] # take the first batch (there is only one)
    
    #print(preds_list)
    return {str(i): preds_list[i] for i in range(10)}

# UI for displaying output class probabilities
output_labels = gr.outputs.Label(num_top_classes=3)

# Main UI that contains everything
interface = gr.Interface(fn=recognize_digit, 
             inputs='sketchpad', 
             outputs=output_labels,
             title='MNIST Drawing Application',
             description='Draw a number 0 through 9 on the sketchpad, and click submit to see the model predictions.',
            )

interface.launch()