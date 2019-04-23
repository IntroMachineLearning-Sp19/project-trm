from fastai.vision import *


def load_images():
    path = Path('data/')
    classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
               'w', 'x', 'y']
    np.random.seed(42)
    data = ImageDataBunch.from_folder(path, train='train', valid='valid', test='test', ds_tfms=get_transforms(),
                                      size=224).normalize(imagenet_stats)
    # data.show_batch(rows=3, figsize=(7,8))
    print(data.classes, data.c, len(data.train_ds), len(data.valid_ds), len(data.test_ds))
    return data


def train_cnn():
    # can use resnet50 for more layers
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(3)
    learn.save('stage-1')

    learn.unfreeze()
    learn.fit_one_cycle(2, max_lr=slice(1e-4, 1e-3))
    learn.save('stage-2')
    return learn


def load_cnn():
    path = Path('data/')
    learn = load_learner(path)
    return learn


def do_prediction(learner, image):
    preds = []

    pred_class, pred_idx, outputs = learner.predict(img)
    # Print the predition
    print(pred_class)
    for i in range(0, 30):
        # data.test_ds.x[i], can iterate through this array for test images
        p = learner.predict(img)
        preds.append(str(p[0]))
        # print(preds)
    return preds


import os

path = Path('data')

if __name__ == "__main__":
    preds = []
    learner = load_cnn()
    for subdir, dirs, files in os.walk(path / 'test'):
        for file in files:
            path = os.path.join(subdir, file)
            img = open_image(path)
            p = do_prediction(learner, img)
            preds.append(str(p[0]))
            print(preds)

    # interp = ClassificationInterpretation.from_learner(learn)
    # interp.plot_confusion_matrix()
    # interp.plot_top_losses(9, figsize=(10,10))

# Testing:
# learner.export()
# This will create a file named 'export.pkl' in the directory 
# where we were working that contains everything we need to deploy 
# our model (the model, the weights but also some metadata like the classes or the transforms/normalization used).

