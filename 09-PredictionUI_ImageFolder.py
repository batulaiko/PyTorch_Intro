import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import gradio as gr

# Weather Image Recognition Dataset
# https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

# DEFINE MODEL (required for loading weights)
class ConvNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #######################
        # Convolutional Part
        #######################
        #print(f'Input dims: {x.shape}')
        
        x = self.conv1(x) # (N, 1, 28, 28) -> (N, 32, 26, 26)
        #print(f'After conv1 {x.shape}')
        x = self.relu(x) # no dim change
        x = self.conv2(x) # (N, 32, 26, 26) -> (N, 64, 24, 24)
        #print(f'After conv2 {x.shape}')
        x = self.relu(x) # no dim change
        x = self.max_pool(x) # (N, 64, 24, 24) -> (N, 64, 12, 12)
        #print(f'After maxpool {x.shape}')
        #######################
        #######################

        #######################
        ## Fully Connected Part
        #######################
        x = torch.flatten(x, 1) # (N, 64, 12, 12) -> (N, 64*12*12) -> (N, 9216)
        x = self.fc1(x) # (N, 9216) -> (N, 128)
        x = self.relu(x) # no dim change
        logits = self.fc2(x) # (N, 128) - (N, 10)
        #######################
        #######################
        
        return logits
        

# LOAD PRE-TRAINED MODEL
model = ConvNet(
    input_channels=3, # 3 for RGB images 
    num_classes=11 # 10 for Weather Image Recognition
)

checkpoint = torch.load('convnet_ImageFolder_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
class_str_name = checkpoint['class_str_names']


# SWITCH MODEL TO PREDICTION ONLY MODE
model.eval()


# Same transforms that used in training
transform=transforms.Compose([
    transforms.Resize((28, 28)), # Resize image into these dimensions
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
    return {class_str_name[i]: preds_list[i] for i in range(10)}


# UI for displaying output class probabilities
output_labels = gr.outputs.Label(num_top_classes=3)


# Main UI that contains everything
interface = gr.Interface(
                fn=recognize_digit, 
                inputs=gr.Image(type='pil'), 
                outputs=output_labels,
                title='Weather Image Recognition Application (ConvNet)',
                description='Upload an Image to make a prediction',
            )


if __name__ == '__main__':
    interface.launch()