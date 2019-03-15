import cv2


class Camera:
    img_counter = 0

    def __init__(self, number, side):
        self.number = number
        self.side = side

    def show_camera(self, r):
        cam = cv2.VideoCapture(self.number)
        self.ret, self.frame = cam.read()
        cv2.namedWindow("test_" + "{}".format(self.side))
        cv2.imshow("test_" + "{}".format(self.side), self.frame)
        if r == False:
            cam.release()
            cv2.destroyAllWindows()

    def capture_image(self):
        self.img_name = "{}".format(self.side) + "_" + "{}.png".format(self.img_counter)
        cv2.imwrite(self.img_name, self.frame)
        self.img_counter += 1


def main():
    camera_list = []
    camera_list.append(Camera(0, "Left"))
    camera_list.append(Camera(1, "Right"))

    while True:
        for camera in camera_list:
            camera.show_camera(True)
            print("{}".format(camera.side) + " camera" + " on!")
        if camera_list[0].ret == None:
            print("Camera not detected")
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            for camera in camera_list:
                camera.capture_image()
            print("New images have been captured and saved!")
    for camera in camera_list:
        camera.show_camera(False)
        print("{}".format(camera.side) + " camera" + " off!")


if __name__ == "__main__":
    main()
