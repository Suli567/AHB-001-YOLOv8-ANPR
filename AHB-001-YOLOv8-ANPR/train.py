from ultralytics import YOLO


def main():
# Load a model
    model = YOLO("yolov8n.pt") # using pre-trained weight

# Use the model
    results = model.train(data = "anpr.yaml", epochs = 50, batch = 10, plots = True)  # Train the model

if __name__ == '__main__':
    main()