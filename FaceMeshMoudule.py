import cv2
import mediapipe as mp
import time

# cap = cv2.VideoCapture("Videos/88.mp4")

class FaceMesh:
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2) # max_num_faces=1 rgb
        self.drawSpec= self.mpDraw.DrawingSpec((255,0,0),thickness=1,circle_radius=2)




    def drawPoints(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          self.drawSpec,self.drawSpec) # FACE_CONNECTIONS
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    print(id, x, y)


def main():
    cap = cv2.VideoCapture(0)
    facemesh = FaceMesh()
    pTime = 0
    while True:
        suceess, img = cap.read()
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        facemesh.drawPoints(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 20), cv2.FONT_HERSHEY_PLAIN,
                    1, (0,255,0),1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()