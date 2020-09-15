import argparse
import cv2
from model import my_model

def getArgument():
    arg = argparse.ArgumentParser()
    # định nghĩa một tham số cần parse
    arg.add_argument('-i', '--image_path',help='link to image')
    # Giúp chúng ta convert các tham số nhận được thành một object và gán nó thành một thuộc tính của một namespace.
    return arg.parse_args()


arg = getArgument()

# đọc được các thuộc tính  từ đường dẫn

img = cv2.imread(arg.image_path)
img = cv2.resize(img,(96,96))

my_model = my_model(False)
#predict hình ảnh
my_model.predict(img)

#show ảnh
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()