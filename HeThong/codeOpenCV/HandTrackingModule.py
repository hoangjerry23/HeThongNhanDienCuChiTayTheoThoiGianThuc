import cv2
import math
import cvzone
from cvzone.HandTrackingModule import HandDetector
import firebase_admin
from firebase_admin import credentials, db
import time

# Initialize Firebase
cred = credentials.Certificate("C:/Users/kazam/Downloads/hand_gesture_control-main/hand_gesture_control-main/test01-41b94-firebase-adminsdk-i7cgp-3a02fa9157.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://test01-41b94-default-rtdb.asia-southeast1.firebasedatabase.app/'
})
ref = db.reference('ESP32_NodeMCU_Broad/Outputs/Digital')

# Initialize Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

light_on = False  # Track LED status
brightness_level = 0  # Track brightness level
last_brightness_level = -1  # Store previous brightness level to check changes

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (640, 480))
    img = cv2.flip(img, 1)  # Flip horizontally

    # Detect hand and get landmark list
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lm_list = hands[0]["lmList"]

        # Kiểm tra trạng thái ngón giữa, ngón áp út và ngón út (gập lại)
        middle_finger_folded = lm_list[12][1] > lm_list[10][1]  # Ngón giữa
        ring_finger_folded = lm_list[16][1] > lm_list[14][1]    # Ngón áp út
        pinky_finger_folded = lm_list[20][1] > lm_list[18][1]   # Ngón út

        # Tắt đèn khi ba ngón đều gập, bật đèn khi có ngón nào không gập
        if middle_finger_folded and ring_finger_folded and pinky_finger_folded and light_on:
            light_on = False
            brightness_level = 0  # Reset brightness to 0
            ref.update({"LED1": False, "DoSang": brightness_level})
            print("LED turned off")

        elif not (middle_finger_folded and ring_finger_folded and pinky_finger_folded) and not light_on:
            light_on = True
            ref.update({"LED1": True})
            print("LED turned on")

        # Điều chỉnh độ sáng khi ngón cái và ngón trỏ chạm nhau
        if light_on:
            x1, y1 = lm_list[4][0], lm_list[4][1]  # Ngón cái
            x2, y2 = lm_list[8][0], lm_list[8][1]  # Ngón trỏ

            # Tính khoảng cách giữa ngón cái và ngón trỏ
            distance = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2))

            # Nếu khoảng cách rất nhỏ (ngón tay chạm nhau), đặt độ sáng về 0
            if distance < 20:  # Tùy chỉnh ngưỡng nếu cần
                brightness_level = 0
            else:
                brightness_level = min(max(int((distance / 150) * 100), 0), 100)

            # Cập nhật độ sáng trong Firebase chỉ khi có sự thay đổi
            if brightness_level != last_brightness_level:
                ref.update({"DoSang": brightness_level})
                last_brightness_level = brightness_level

            # Vẽ vòng tròn và đường kết nối giữa ngón cái và ngón trỏ
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)  # Vòng tròn đỏ trên ngón cái
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)  # Vòng tròn đỏ trên ngón trỏ
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)       # Đường màu xanh lá cây

            # Tính toán điểm trung tâm
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # Vòng tròn xanh dương ở trung tâm

    # Display LED status and brightness level
    cv2.putText(img, f"LED Status: {'ON' if light_on else 'OFF'}", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(img, f"Brightness: {brightness_level}%", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
