import numpy as np
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train_input_dir = "fer2013/train"
test_input_dir = "fer2013/test"
categories = [
  "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]


def numeric_images(input_dir, categories):
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

  return (datas, target)


x_train, y_train = numeric_images(train_input_dir, categories)
x_test, y_test = numeric_images(test_input_dir, categories)

model = RandomForestClassifier()
parameters = [{
  "n_estimators": [100, 200, 500, 1000],
  "max_depth": [5, 10, 20],
  "min_samples_split": [2, 5, 10],
  "min_samples_leaf": [1, 2, 5],
  "max_features": ["sqrt", "log2"]
}]

grid_search = GridSearchCV(model, parameters)

grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

y_predicted = best_estimator.predict(x_test)

score = accuracy_score(y_test, y_predicted)

print('Score of model: {}'.format(str(score)))
