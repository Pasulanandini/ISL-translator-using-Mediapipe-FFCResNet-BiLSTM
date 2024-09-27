import math
import numpy as np
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def cropping2(image):
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    # Run MediaPipe Hands.
    with mp_hands.Hands(
         static_image_mode=False,
         max_num_hands=2,
         min_detection_confidence=0.05,
         min_tracking_confidence=0.05) as hands:
        
            # Convert the BGR image to RGB, flip the image around y-axis for correct 
            # handedness output and process it with MediaPipe Hands.
            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            # Draw hand landmarks of each hand.
            image_height, image_width, _ = image.shape
            annotated_image = cv2.flip(image.copy(), 1)
            image=cv2.flip(image,1)
            background=cv2.imread(r'C:\Users\JNTU HYD\Downloads\livefiles\livefiles\white.png')
            background=cv2.resize(background,(1080,1920))
            i=0
            if(results.multi_hand_landmarks==None):
                return background
            for hand_landmarks in results.multi_hand_landmarks:
                tip1=(int(0.02*image_width+max(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width,
                )),int(0.02*image_height+max(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height,
                )))
                tip2=(int(-0.025*image_width+min(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width,
                )),int(-0.025*image_height+min(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height,    
                )))
                color = (255, 0, 0) 
                # Line thickness of 2 px 
                thickness = 10
                # Using cv2.rectangle() method 
                # Draw a rectangle with blue line borders of thickness of 2 px 
      
                #image = cv2.rectangle(image, tip2, tip1, color, thickness)
                mask = np.zeros(image.shape[:2], dtype="uint8")
             
                
                if(i==0):
                    mask1=cv2.rectangle(mask, tip1, tip2, 255, -1) 
                    mask2=np.zeros(image.shape[:2], dtype="uint8")
                    tipminx=min(tip1[0],tip2[0])
                    tipmaxx=max(tip1[0],tip2[0])
                    tipminy=min(tip1[1],tip2[1])
                    tipmaxy=max(tip1[1],tip2[1])
                else:
                    mask2=cv2.rectangle(mask, tip1, tip2, 255, -1)
                    tipminx=min(tip1[0],tip2[0],tipminx)
                    tipmaxx=max(tip1[0],tip2[0],tipmaxx)
                    tipminy=min(tip1[1],tip2[1],tipminy)
                    tipmaxy=max(tip1[1],tip2[1],tipmaxy)
                i=1
                mp_drawing.draw_landmarks(
                    background,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                #cv2.imshow('abcd',cv2.flip(annotated_image, 1))
                #cv2.waitKey(0)
# apply our mask -- notice how only the person in the image is
# cropped out
            
            #cv2.imshow('abcd',masked)
            #cv2.waitKey(0)
            #mask=cv2.bitwise_or(mask1,mask2)
            #masked = cv2.bitwise_and(image, image, mask=mask)
            output=cv2.flip(background,1)
        
    
            #output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            #output=cv2.resize(output,(299,299))
    
    return output

if __name__ == '__main__':
    frame = cv2.imread("test.jpeg")
    cropping2(frame)
