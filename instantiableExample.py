from instantiableObjectDetection import balloonDetectionModel
import cv2

weight_path = './modelPW.pth'
input_path = './archive/Balloon/valid/images/485227412_e335662bb5_b_jpg.rf.6e089c2cb112153d1e17c27993d6b71e.jpg'

model = balloonDetectionModel(weight_path,input_path)
#output: output[0] is best box in format [x_min,y_min,x_max,y_max]; output[1] is best center in format [x,y]
#Note: output coordinates are based on a resize of the image to 320x320
output = model.analyzeImageCentered(0.5)

#debugging and visualization
img = cv2.imread(input_path)
img = cv2.resize(img,(320,320))
cv2.circle(img,center=(output[1][0],output[1][1]),radius = 5,color=(0,255,0),thickness=-1)
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
