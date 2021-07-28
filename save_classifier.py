import torchvision.models as models
import mlflow.pytorch
import torch
from torchvision import transforms
import PIL


with mlflow.start_run() as run:
    # Actual training loop would have come here
    normalize = transforms.Compose(  # noqa
        [transforms.ToTensor(),  # noqa
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]  # noqa
    )
    image = normalize(PIL.Image.open('fish.jpg'))
    model = models.resnet18(pretrained=True).eval()
    out = model(image.unsqueeze(0))
    scripted_model = torch.jit.script(model)
    print("Saving TorchScript Model ..")
    mlflow.pytorch.log_model(scripted_model, artifact_path='classifier')
