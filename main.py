import numpy as np
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

input_dir = "fer2013"
categories = [
  "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

datas = []
target = []

target_size = (48, 48)

for index, category in enumerate(categories):
  for file in os.listdir(os.path.join(input_dir, category)):
    img_path = os.path.join(input_dir, category, file)
    image = Image.open(img_path).convert("L")
    image = image.resize(target_size)
    datas.append(list(image.getdata()))
    target.append(index)

datas = np.asarray(datas)
target = np.asarray(target)

model = RandomForestClassifier()
parameters = [{
  "n_estimators": [100, 200, 500, 1000],
  "max_depth": [5, 10, 20],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 5],
  "max_features": ["sqrt", "log2"]
}]

grid_search = GridSearchCV(model, parameters)

grid_search.fit(datas, target)

best_estimator = grid_search.best_estimator_
