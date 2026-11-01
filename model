import os
import torch

from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.transforms import v2 as T

import matplotlib.pyplot as plt

import time

import cv2

from torchcodec.decoders import decode_image



class ObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root, subfile, transforms=None): #root, train/valid, transforms
        self.root = root
        self.subfile = subfile
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned

        self.imgs = list(sorted(os.listdir(os.path.join(root, "Balloon", subfile, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Balloon", subfile, "labels"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Balloon", self.subfile, "images", self.imgs[idx])
        box_path = os.path.join(self.root, "Balloon", self.subfile, "labels", self.masks[idx])
        img = decode_image(img_path)
        img = F.resize(img, [320,320]) #resize image to 320x320
        img_x, img_y = img.shape[1:] #get image x and y

        box = []
        with open(box_path) as f: #cord: x_center, y_center, width, height
            text = f.readlines()
            for i in range(len(text)):
                text[i] = text[i].strip() #removes extra spaces
                text[i] = text[i].replace("\n","") #removed "\n"
                text[i] = text[i].split(" ") #splits on space
                del text[i][0] #object id is always 0 since there is one class
                #translates the percentages given to pixel values
                x_center = float(text[i][0])*img_x
                y_center = float(text[i][1])*img_y
                width = float(text[i][2])*img_x
                height = float(text[i][3])*img_y
                #translate the given xcycwh to xyxy
                x_min = int(x_center - (width/2))
                y_min = int(y_center - (height/2))
                x_max = int(x_center + (width/2))
                y_max = int(y_center + (height/2))
                box.append([x_min,y_min,x_max,y_max])
        collate_amount = 1 #defines how long the minimum box length should be
        if len(box)<collate_amount: #pads so every box list have at least one box
            while len(box)<collate_amount:
                box.append([1,1,2,2])
        boxes = torch.as_tensor(box)

        num_objs = len(boxes)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = boxes[:, 3] * boxes[:, 2]

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {"boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
                   "labels": labels, "image_id": image_id, "area": area}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    #model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

# train on the accelerator or on the CPU, if an accelerator is not available
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
print(device)
# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
dataset = ObjectDetectionDataset('archive','train', get_transform(train=True))
dataset_test = ObjectDetectionDataset('archive','valid', get_transform(train=False))

def custom_collate_fn(batch):
    return batch

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=custom_collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=True,
    collate_fn=custom_collate_fn
)

def to_cuda(data):
    if isinstance(data, torch.Tensor):
        return data.to("cuda")
    elif isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cuda(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_cuda(v) for v in data)
    return data


# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,
    gamma=0.1
)

num_epochs = 0

criterion = torch.nn.CrossEntropyLoss()
model.train()

lc = [] #loss_classifier (how accurately the model identifies what is in the box)
lbr = [] #loss_box_reg  (measures tight fit of the bounding boxes(bounding box regression))
lo = [] #loss_objectness (measures how well the Region Proposal Network(RPN) distinguishes between foreground and background
lrbr = [] # loss_rpn_box_reg (Measures accuracy of the RPN's rough, initial bounding box proposals)
tl = []
pretime = time.time()
count = 0
torch.autograd.set_detect_anomaly(True)

print("Starting training...")
for epoch in range(num_epochs):

    for data in data_loader: #data is list with a tensor and a dictionary
        optimizer.zero_grad() #
        inputs = []
        targets = []
        for d in data: #adds all tensors(images) to inputs and all dictionaries(targets) to targets
            inputs.append(d[0])
            targets.append(d[1])
        inputs = to_cuda(inputs) #sets the inputted data to run on device
        targets = to_cuda(targets) #sets the inputted data to run on device. Assets in setting a dictionary to device as there is no library method for such
        output = model(inputs,  targets) #forward

        total_loss = sum(loss for loss in output.values()) #added losses

        #optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #prevents overflow
        optimizer.step()

        #saves losses
        lc.append(output["loss_classifier"].item())
        lbr.append(output["loss_box_reg"].item())
        lo.append(output["loss_objectness"].item())
        lrbr.append(output["loss_rpn_box_reg"].item())
        tl.append(total_loss.item())

        #progress and bug catching print
        print("Epoch: "+str(epoch+1)+"/"+str(num_epochs)+" Num: "+str(count)+"/2256 "
                "Current total loss: "+str(total_loss.item())+
                " Classifier: "+str(output["loss_classifier"].item())+
                " Box Regression: "+str(output["loss_box_reg"].item())+
                " Objectness: "+str(output["loss_objectness"].item())+
                " Initial Box: "+str(output["loss_rpn_box_reg"].item()))

        count += 1 #tracks which batch the model is training on
    count = 0
    lr_scheduler.step() #lowers learning rate for finetuning

    #Bug catching prints
    #print("\nEpoch: "+str(epoch)+" finished.")
    #print("Time since start: "+str(time.time()-pretime)+" seconds.")
    #print("Classifier Loss: "+str(lc[len(lc)-1])+" Loss Difference since start: "+str(lo[len(lc)-1]-lc[0]))
    #print("Bounding Box Regression Loss: "+str(lbr[len(lbr)-1])+" Loss Difference since start: "+str(lbr[len(lbr)-1]-lbr[0]))
    #print("Foreground-Background Loss: "+str(lo[len(lo)-1])+" Loss Difference since start: "+str(lo[len(lo)-1]-lo[0]))
    #print("Initial Box Proposal Loss: "+str(lrbr[len(lc)-1])+" Loss Difference since start: "+str(lo[len(lc)-1]-lrbr[0]))
    #print("Total Loss: "+str(tl[len(tl)-1])+" Loss Difference since start: "+str(tl[len(tl)-1]-tl[0])+"\n")

print("Training Finished")

#graph of losses over time
plt.plot(lc,color='#ff0000', label='Loss Classifier')
plt.plot(lbr,color='#00ff00', label='Loss Bounding Box Regression')
plt.plot(lo,color='#0000ff', label='Loss Objectness')
plt.plot(lrbr,color='#ff00ff', label='Loss Initial Box Proposal')
plt.plot(tl,color='#ff8000', label='Combined Loss')
plt.title("Loss over time")
plt.xlabel("Batches")
plt.ylabel("Loss")
plt.legend()
plt.show()

#torch.save(model.state_dict(), './modelPW.pth') #saves model weights(Note: only saves weights)
imported_model = get_model_instance_segmentation(num_classes)# new model for imported weights
imported_model.load_state_dict(torch.load('./modelPW.pth', weights_only=True))#set new model weights to imported
#print(torch.load('./model1.pth', weights_only=True)) #debugging

imported_model.eval() #so the model doesn't train on this
for j in range(4): #Shows model prediction of first j validate images
    data = next(iter(data_loader_test)) #data is a list with a tensor and dictionary
    inputs = []
    targets = []
    inputs.append(data[0][0])#add the tensor to its own list
    targets.append(data[0][1])#add the dictionary to its own list

    image = inputs[0] #gets image for display purposes
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)#turns the 0-1 that the tensor is in into 0-255
    image = image[:3, ...] #gets the last 3 dimensions
    img = image.permute(1,2,0).detach().cpu().numpy() #makes a copy of the image, makes it a numpy array, and reorders it so that csv can use it

    output = imported_model(inputs) #gets model prediction
    boxes = output[0]["boxes"] #gets just the box predictions from output

    for i in range(len(output[0]["boxes"])): #itterates the boxes
        #Separates the box coordinates
        x_min, y_min, x_max, y_max = output[0]["boxes"][i]
        x_min = int(x_min.item())
        y_min = int(y_min.item())
        x_max = int(x_max.item())
        y_max = int(y_max.item())
        if output[0]["scores"][i] >.5: #if the model is more than .5 confident (more than 50% confident)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3) #adds rectangle to image
    cv2.imshow("Image", img) #shows validate image with predicted rectangles
    cv2.waitKey(0)
    cv2.destroyAllWindows()


