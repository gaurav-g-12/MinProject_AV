import cv2
import numpy as np

path = r'D:\5th sem\ML\Data\35.mp4'
path_save = r'D:\5th sem\ML\Data_images\Pans'
count = 0
cap = cv2.VideoCapture(path)

while(1):
    _, image = cap.read()
    count = count+1
    filename = path_save+'\\'+'g'+str(count)+'.jpg'
    cv2.imwrite(filename, image)

    if cv2.waitKey(0) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()