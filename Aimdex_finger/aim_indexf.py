import mediapipe as mp
import cv2
import random
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
score = 0

x_enemy = random.randint(50, 600)
y_enemy = random.randint(50, 400)

def enemy(image):
    global score, x_enemy, y_enemy
    cv2.circle(image, (x_enemy, y_enemy), 25, (0, 0, 255), 5)

video = cv2.VideoCapture(0)

start_time = time.time()
time_limit = 30  

with mp_hands.Hands(max_num_hands = 1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        _, frame = video.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)

        image_height, image_width, _ = image.shape

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX
        score_color = (0, 255, 0)  
        time_color = (255, 255, 0)    
        score_position = (600, 30)     

        cv2.putText(image, "Score: ", (480, 30), font, 1, score_color, 2, cv2.LINE_AA)
        cv2.putText(image, str(score), score_position, font, 1, score_color, 2, cv2.LINE_AA)

        remaining_time = int(time_limit - (time.time() - start_time))
        if remaining_time <= 0:
            cv2.putText(image, "Time's Up!", (125, 250), font, 2, (0, 0, 255), 5, cv2.LINE_AA)
        else:
            cv2.putText(image, f"Time: {remaining_time}s", (480, 100), font, 1, time_color, 2, cv2.LINE_AA)

        enemy(image)

        if results.multi_hand_landmarks != None:
            for hand_landmarks in results.multi_hand_landmarks:
                for point in mp_hands.HandLandmark:
                    normalized_landmark = hand_landmarks.landmark[point]
                    pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(
                        normalized_landmark.x, normalized_landmark.y, image_width, image_height)

                    if point == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                        try:
                            cv2.circle(image, (pixel_coordinates_landmark[0], pixel_coordinates_landmark[1]), 25, (0, 200, 0), 5)

                            if abs(pixel_coordinates_landmark[0] - x_enemy) < 25 and abs(pixel_coordinates_landmark[1] - y_enemy) < 25:
                                x_enemy = random.randint(50, 600)
                                y_enemy = random.randint(50, 400)
                                score += 1
                        except:
                            pass

        if remaining_time <= 0:
            cv2.putText(image, "Game's Over!", (100, 350), font, 2, (0, 0, 255), 5, cv2.LINE_AA)

        cv2.imshow("Hand Tracking Device", image)

        if cv2.waitKey(10) & 0xFF == ord("q") or remaining_time <= 0:
            print("Final score:", score)
            time.sleep(2)
            break

video.release()
cv2.destroyAllWindows()