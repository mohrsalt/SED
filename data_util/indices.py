from audioset_classes import as_strong_train_classes

from audioset_classes import as_danger_classes

# print(len(as_strong_train_classes))

for danger_class in as_danger_classes:
    print(as_strong_train_classes.index(danger_class))