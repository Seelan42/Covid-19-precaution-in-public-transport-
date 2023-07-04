# Covid-19-precaution-in-public-transport










import cv2
import numpy as np

# Load the thermal imaging camera
cap = cv2.VideoCapture(0)

# Set the thermal imaging camera to a higher temperature range
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))

while True:
    # Read the thermal image from the camera
    _, thermal_image = cap.read()

    # Convert the thermal image to grayscale
    gray_image = cv2.cvtColor(thermal_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the thermal image
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(thermal_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the temperature data at the location of the face
        face_temperature = thermal_image[y:y+h, x:x+w].mean()

        # Check if the temperature is within normal range
        if face_temperature < 37.5 or face_temperature > 38:
            cv2.putText(thermal_image, "Abnormal Temperature", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Extract the face ROI
        face_roi = gray_image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (150, 150))

        # Detect masks in the face ROI
        masks = mask_cascade.detectMultiScale(face_roi, 1.1, 4)

        # Check if a mask is detected
        if len(masks) == 0:
            cv2.putText(thermal_image, "No Mask", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(thermal_image, "Mask", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the thermal image
    cv2.imshow("Thermal Imaging", thermal_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the thermal imaging camera
cap.release()

# Close all windows
cv2.destroyAllWindows()
