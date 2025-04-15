from ultralytics import YOLO

if __name__ == '__main__':

  model=YOLO('LEPS.yaml')
  results = model.train(data='pothole.yaml')