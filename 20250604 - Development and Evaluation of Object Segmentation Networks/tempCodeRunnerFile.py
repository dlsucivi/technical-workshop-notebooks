ONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y), color, -1)
    cv2.putText(frame, label, (x, y - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

# Start webcam capture
cap = cv2.VideoCapture(0)