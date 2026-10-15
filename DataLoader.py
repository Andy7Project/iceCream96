import zipfile
import numpy as np
import cv2

currentImageIndex = 0 #[1,2235]
image_x_size = 640
image_y_size = 640
root_path = "./archive.zip"
training_img_path  = "Balloon/train/images/"
training_label_path = "Balloon/train/labels/"
validation_img_path = "Balloon/validation/images/"
validation_label_path = "Balloon/validation/labels/"
change = 0

def parse_yolo(yolo_cords, image):
    min_max_cords = []
    #print(yolo_cords)
    if len(yolo_cords)>0 and yolo_cords[0][0]!='':
        for cord in yolo_cords:
            str_cord = [float(x) for x in cord]
            x_center, y_center, width, height = str_cord
            x_min = int((x_center * image_x_size) - ((width * image_x_size) / 2))
            y_min = int((y_center * image_y_size) - ((height * image_y_size) / 2))
            x_max = int((x_center * image_x_size) + ((width * image_x_size) / 2))
            y_max = int((y_center * image_y_size) + ((height * image_y_size) / 2))
            min_max_cords.append([x_min, y_min, x_max, y_max])
    return min_max_cords

def retrieve_image(file_name, zip_ref):
    formatted_raw = np.frombuffer(zip_ref.open(file_name).read(), dtype=np.uint8)
    image = cv2.imdecode(formatted_raw, cv2.IMREAD_COLOR)
    return image

def retrieve_cords(file_name, file_list, zip_ref):
    file_id = file_name[len(training_img_path):len(file_name) - 4] + ".txt"
    #iterates through and gets the cords with a matching name to the file
    temp = next((name for name in file_list if name.endswith(file_id)), None)
    cords = zip_ref.open(temp).read().decode("utf-8")
    cords = cords[1:].split("\n0")
    total_cords = []
    for cord in cords:
        total_cords.append(cord[1:].split(" "))
    return total_cords

def display_image(image, min_max_cords):
    if len(min_max_cords)>0 and min_max_cords[0][0]!='':
        for cord in min_max_cords:
            x_min, y_min, x_max, y_max = cord
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2256 images in training
def retrieve_batch(batch_size=128):

    #with root_path.open(mode = 'r') as zip_ref:
    with zipfile.ZipFile("./archive.zip", 'r') as zip_ref:
        global currentImageIndex
        list_continues = True
        zip_list = zip_ref.namelist()
        train_img_list = [f for f in zip_list if f.startswith(training_img_path)]
        #print(len(train_img_list))
        train_label_list = [f for f in zip_list if f.startswith(training_label_path)]
        adjusted_batch_size = min(batch_size,abs(len(train_img_list)-currentImageIndex))
        if abs(len(train_img_list)-currentImageIndex)<batch_size:
            list_continues = False
        img_list = []
        label_list = []
        #print(adjusted_batch_size)
        for i in range(adjusted_batch_size):
            file_name = train_img_list[i+currentImageIndex]
            #print(file_name)
            yolo_cords = retrieve_cords(file_name, train_label_list, zip_ref)
            #print(yolo_cords)
            image = retrieve_image(file_name, zip_ref)
            min_max_cords = parse_yolo(yolo_cords, image)
            img_list.append(image)
            label_list.append(min_max_cords)
            #minMaxCords
            #display_image(image, min_max_cords)
            #return image, min_max_cords
            #zip_ref.close()
        currentImageIndex += batch_size
        return img_list, label_list, list_continues

continues =True
batch_num = 0
completed = 0
while continues:
    img_list, labels_list, continues = retrieve_batch()
    batch_num+= 1
    completed += len(img_list)
    print("Batch: " +str(batch_num))
    print("Completed: " +str(len(img_list)))
    print(str(completed)+" / "+str(2256))




#retrieve_batch()
