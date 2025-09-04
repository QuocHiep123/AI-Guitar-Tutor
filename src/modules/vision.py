import cv2
import mediapipe as mp

# -- khởi tạo Miediapipe --
mp_hands = mp.solutions.hands
# Tham số trong Hands():
# static_image_mode=False: Xử lý video stream.
# max_num_hands=1: Chỉ nhận diện 1 bàn tay để tối ưu hiệu suất.
# min_detection_confidence=0.7: Độ tin cậy tối thiểu để coi là một bàn tay.
hand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils # tiện ích vẽ các điểm trên bàn tay

cap = cv2.VideoCapture(0) # Mở camera

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể kết nối camera")
        break
    # lật ảnh để dễ điều khiển hơn
    frame = cv2.flip(frame, 1)
    #1. Chuyển màu sang RGB vì cv2 xài BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #2. Tìm bàn tay
    results = hand.process(rgb_frame)
    
    # vẽ landmarks
    if results.multi_hand_landmarks:
        # lặp qua từng bàn tay (mình đang dùng max = 1)
        for hand_landmarks in results.multi_hand_landmarks:
            # vẽ các điểm landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    cv2.imshow("AI GUITAR TUTOR - LESSON 1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print('Đóng chương trình')
cap.release()
cv2.destroyAllWindows()
    