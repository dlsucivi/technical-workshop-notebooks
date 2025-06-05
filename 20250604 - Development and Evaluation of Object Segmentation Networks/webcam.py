import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50

# ====== SETTINGS ======
MODEL_PATH = "checkpoints/best_model.pth"
NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [ #ORDERED the same way the IDs are ordered in MS COCO
    "person", "dog", "cat", "cup", "fork", "knife", "spoon", "chair", "mouse", "remote", "keyboard", "cell phone"
]

# Segmentation Mask colors for visualization
CLASS_COLORS = [
    (128, 64, 128),   # dog
    (244, 35, 232),   # cat
    (70, 70, 70),     # person
    (102, 102, 156),  # chair
    (190, 153, 153),  # mouse
    (153, 153, 153),  # remote
    (250, 170, 30),   # keyboard
    (220, 220, 0),    # cell phone
    (107, 142, 35),   # cup
    (152, 251, 152),  # fork
    (70, 130, 180),   # knife
    (220, 20, 60),    # spoon
]

INFERENCE_FRACTION = 0.1  # Only run on 25% of frames
# =======================

# Define transform to preprocess frames in webcam the same way as training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

# Load the model
def load_model(path):
    model = fcn_resnet50(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model(MODEL_PATH)

# Overlay label text with background box
def draw_label(frame, label, x, y, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    cv2.rectangle(frame, (x, y - text_h - 5), (x + text_w, y), color, -1)
    cv2.putText(frame, label, (x, y - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

# Start webcam capture
cap = cv2.VideoCapture(0)

frame_counter = 0
last_mask = None  # To store the last prediction

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]
        frame_counter += 1

        # Run inference every 1/INFERENCE_FRACTION of frames
        run_inference = (frame_counter % int(1 / INFERENCE_FRACTION) == 0)
        if run_inference:
            input_tensor = transform(frame).unsqueeze(0).to(DEVICE)
            output = model(input_tensor)['out']
            probs = F.softmax(output, dim=1)[0]
            pred_mask = torch.argmax(probs, dim=0).cpu().numpy()
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            last_mask = pred_mask

        # Use the last known mask if inference not run
        overlay = frame.copy()
        alpha = 0.5
        if last_mask is not None:
            for class_idx in np.unique(last_mask):
                if class_idx == 0:
                    continue  # Skip background
                if class_idx >= len(CLASS_NAMES) + 1:   
                    continue
                mask = last_mask == class_idx
                color = CLASS_COLORS[class_idx - 1]  # Background is class 0
                overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

                # Label position
                yx = np.argwhere(mask)
                if yx.size > 0:
                    y, x = yx[0]
                    draw_label(overlay, CLASS_NAMES[class_idx - 1], x, y, color=color)

        cv2.imshow("Segmentation", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
