import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchcodec.decoders import decode_image

# Instantiate the model as its type (should work for any type of torch fasterrcnn)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
#2 classes since the model needs to detect: Background(Class 0) and Balloon(Class 1)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

transform = T.Compose([T.ToDtype(torch.float, scale=True),T.ToTensor()])

class balloonDetectionModel:
    def __init__(self, model_weight_path, input_path):
        self.model = model
        self.model.load_state_dict(torch.load(model_weight_path,weights_only=True))
        self.model.eval()
        self.input_path = input_path

    def analyzeImage(self,confidence_score):
        image_size = 320
        img = decode_image(self.input_path)
        img = F.resize(img,[image_size,image_size])
        img_tensor = transform(img)
        batch = []
        batch.append(img_tensor)

        with torch.no_grad():
            output = model(batch)
        boxes = []
        centers = []
        for i in range(len(output[0]["boxes"])):
            x_min, y_min, x_max, y_max = output[0]["boxes"][i]
            x_min = int(x_min.item())
            y_min = int(y_min.item())
            x_max = int(x_max.item())
            y_max = int(y_max.item())
            if output[0]["scores"][i] >confidence_score: #if the model is more than .5 confident (more than 50% confident)
                boxes.append([x_min,y_min,x_max,y_max])
                x_center = int(x_min+((x_max-x_min)/2))
                y_center = int(y_min+((y_max-y_min)/2))
                centers.append([x_center,y_center])
        return boxes, centers

    def analyzeImageCentered(self,confidence_score): #returns (x,y) of balloon closest to center
        boxes, centers = self.analyzeImage(confidence_score)
        best_box = []
        best_center = [None,None]
        best_dist = 1000000
        for i in range(len(centers)):
            dist = abs(centers[i][0]-320)+abs(centers[i][1]-320)
            if dist<best_dist:
                best_box = boxes[i]
                best_center = centers[i]
                best_dist = dist
        return best_box, best_center






