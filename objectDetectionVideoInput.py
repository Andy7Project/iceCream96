import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
model.load_state_dict(torch.load('./modelPW.pth', weights_only=True))#set new model weights to imported

transform = T.Compose([T.ToDtype(torch.float, scale=True),T.ToTensor()])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Press q to quit.")

model.eval()

while True:
    ret, frame = cap.read()
    if not ret:
        raise IOError("Failed to grab frame")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_tensor = transform(rgb_frame)
    batch = []
    batch.append(input_tensor)

    with torch.no_grad():
        output = model(batch)

    for i in range(len(output[0]["boxes"])): #itterates the boxes
        #Separates the box coordinates
        x_min, y_min, x_max, y_max = output[0]["boxes"][i]
        x_min = int(x_min.item())
        y_min = int(y_min.item())
        x_max = int(x_max.item())
        y_max = int(y_max.item())
        if output[0]["scores"][i] >.5: #if the model is more than .5 confident (more than 50% confident)
            cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3) #adds rectangle to image
    cv2.imshow("Image", rgb_frame) #shows validate image with predicted rectangles

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
