import torch
from torchvision import transforms
from PIL import Image

def process_image(image_path):
    # Define the transformation pipeline for the input image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open and preprocess the image
    image = Image.open(image_path)
    image = transform(image)
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    return image

def predict(image_path, model, topk=5):
    # Process the image
    image = process_image(image_path)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the image tensor to the same device as the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model = model.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities, indices = torch.topk(torch.softmax(output, dim=1), topk)
    
    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.idx_to_class.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    
    # Convert tensor probabilities to list
    probabilities = probabilities.squeeze().cpu().numpy().tolist()
    
    return probabilities, classes

# Example usage:
image_path = "flower.jpg"
probs, classes = predict(image_path, loaded_model)
print(probs)
print(classes)
