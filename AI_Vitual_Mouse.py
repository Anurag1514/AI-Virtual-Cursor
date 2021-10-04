import cv2
import numpy as np
import Hand_Tracking_Module as htm
import autopy

####################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
import time

####################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(3, hCam)
detector = htm.handDetector(detectionCon=0.7, maxHands=1)
wScreen, hScreen = autopy.screen.size()


while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert Co-ordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScreen))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScreen-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both index and middle finger are up it is in clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            # 9. Find the distance b/w fingers
            length, img, infoLine = detector.findDistance(8, 12, img)

            # 10. Click Mouse if distance short
            if length < 40:
                cv2.circle(img, (infoLine[4], infoLine[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # 11. Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 12. Display
    cv2.imshow("Virtual_Mouse", img)
    cv2.waitKey(1)
