import gradio as gr
from torchvision import models, transforms
from PIL import Image
import torch, os

model = models.mobilenet_v2(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

labels = open(os.path.join(os.path.dirname(__file__), "labels.txt")).read().splitlines()

def classify_image(image):
    input_tensor = transform(Image.fromarray(image)).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(input_tensor)[0].softmax(0)
    top5 = probabilities.topk(5)
    return {labels[catid]: prob.item() for prob, catid in zip(*top5)}

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=5),
    title="Classify animals",
    examples=[[os.path.join(os.path.dirname(__file__), "cheetah.jpg")]],
).launch()