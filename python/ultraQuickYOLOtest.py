from ultralytics import YOLO

# ---------------------------------------------------------

# Takes a video, apply a YOLO model, and display the result with prebuilt functions

# ---------------------------------------------------------

# model = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/train640_32_500_manuel/weights/best.pt")
model = YOLO("C:/Users/blagn771/Documents/Aquaman/Aquaman/runs/segment/bestProjet1a.pt")
path = "C:/Users/blagn771/Downloads/000000 (2) - Trim.mp4"

model.predict(path, show=True, save=True)