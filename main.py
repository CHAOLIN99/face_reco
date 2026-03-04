import face_recognition
import cv2
import numpy as np
import os
from pathlib import Path

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        
    def load_known_faces(self, known_faces_dir="data/known_faces"):
        """Load and encode known faces from directory"""
        print("Loading known faces...")
        
        for filename in os.listdir(known_faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Load image
                image_path = os.path.join(known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    # Use filename without extension as name
                    name = os.path.splitext(filename)[0]
                    self.known_face_names.append(name)
                    print(f"✓ Loaded: {name}")
                else:
                    print(f"✗ No face found in: {filename}")
        
        print(f"\nTotal faces loaded: {len(self.known_face_names)}")
    
    def recognize_faces_in_image(self, image_path, output_dir="output"):
        """Recognize faces in a single image"""
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Find faces and their encodings
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Convert to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if face matches known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with smallest distance
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            # Draw rectangle and name
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Save result
        output_path = os.path.join(output_dir, f"result_{Path(image_path).name}")
        cv2.imwrite(output_path, image)
        print(f"Result saved to: {output_path}")
        
        return image
    
    def recognize_from_webcam(self):
        """Real-time face recognition from webcam"""
        video_capture = cv2.VideoCapture(0)
        
        print("\nStarting webcam... Press 'q' to quit")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Resize for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Check matches
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                
                # Draw on frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

def main():
    # Initialize system
    fr_system = FaceRecognitionSystem()
    
    # Load known faces
    fr_system.load_known_faces()
    
    print("\n--- Face Recognition System ---")
    print("1. Recognize faces in image")
    print("2. Real-time webcam recognition")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        image_path = input("Enter path to image: ")
        fr_system.recognize_faces_in_image(image_path)
    elif choice == "2":
        fr_system.recognize_from_webcam()
    else:
        print("Goodbye!")

if __name__ == "__main__":
    main()