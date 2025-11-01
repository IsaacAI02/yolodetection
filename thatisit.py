from picamera2 import Picamera2
import cv2
from ultralytics import YOLO
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 120)
engine.setProperty('volume', 1.0)

def speak(text):
    
    time.sleep(0.2)
    engine.say(text)
    engine.runAndWait()
    

def format_price(detect_string):
    if len(detect_string) >= 3 and detect_string[-1] == detect_string[-2]:
        return detect_string[:-2] + "." + detect_string[-2:]
    
    return detect_string
    
    

# Define your 11 classes: 0–9 and dot
class_names = ['.','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Load your trained YOLO model
model = YOLO("/home/isaackun/Desktop/best.pt")  # Replace with your path

# Initialize Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (480, 640)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
speak("the program started")
last_price = None
frame_count = 0
while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # Run YOLO inference
    if frame_count%10 == 0:
        results = model.predict(source=frame, conf=0.5, device="cpu", imgsz=320, verbose=False)[0]

    digits = []

    # Loop through detections
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label = class_names[cls_id]
        center_x = (x1 + x2) // 2
        digits.append((center_x, label))

        # Draw boxes (optional)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Sort digits by horizontal position
    digits.sort(key=lambda x: x[0])
    price_str = "".join([d[1] for d in digits]) if digits else ""

    # Show full price on screen
    if price_str:
        price_str = format_price(price_str)
        cv2.putText(frame, f"Price: {price_str}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        if last_price != price_str:
            speak(price_str)
            time.sleep(0.1)
            last_price = price_str
        

    # Display frame
    cv2.imshow("YOLO Price Reader", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
