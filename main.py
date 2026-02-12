import cv2
import os
import time
import HandTrackingModule as htm
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    ptime = 0
    cap.set(3, 640)
    cap.set(4, 480)

    fingerpath = "fingers"

    if not os.path.exists(fingerpath):
        print("Error: 'fingers' folder not found")
        return

    mylist = sorted(os.listdir(fingerpath))
    overlayList = []

    for impath in mylist:
        image = cv2.imread(os.path.join(fingerpath, impath))
        if image is not None:
            overlayList.append(image)

    detector = htm.HandDetector(detectionCon=0.7)
    tipid = [4, 8, 12, 16, 20]

    while True:
        success, img = cap.read()
        if not success:
            print("Frame capture failed")
            break

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)
        handType = detector.getHandType()

        if len(lmList) != 0:
            finger = []

            if handType == "Right":
                if lmList[tipid[0]][1] < lmList[tipid[0] - 1][1]:
                    finger.append(1)
                else:
                    finger.append(0)
            else:
                if lmList[tipid[0]][1] > lmList[tipid[0] - 1][1]:
                    finger.append(1)
                else:
                    finger.append(0)

            for id in range(1, 5):
                if lmList[tipid[id]][2] < lmList[tipid[id] - 2][2]:
                    finger.append(1)
                else:
                    finger.append(0)

            totalfingers = finger.count(1)

            if 0 < totalfingers <= len(overlayList):
                overlay = cv2.resize(overlayList[totalfingers - 1], (200, 200))
                h, w, _ = overlay.shape
                img[0:h, 0:w] = overlay

            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalfingers), (45, 375),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 25)

            if bbox:
                x, y, w, h = bbox
                text_y = y - 10 if y - 10 > 20 else y + 30
                cv2.putText(img, handType, (x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ctime = time.time()
        fps = 1 / (ctime - ptime) if ctime != ptime else 0
        ptime = ctime

        cv2.putText(img, "FPS: " + str(int(fps)), (500, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        cv2.imshow("Finger Counter (Both Hands)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
