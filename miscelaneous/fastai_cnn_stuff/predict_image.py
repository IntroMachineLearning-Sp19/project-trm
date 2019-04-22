from fastai.vision import *
folder = 'a'
path = Path('data/')
dest = path/folder
classes = ['a','b','c', 'd', 'e', 'f', 'g', 'h', 'i','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y']
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train='train', valid='valid', test='test',ds_tfms=get_transforms(), size=100).normalize(imagenet_stats)
# data.show_batch(rows=3, figsize=(7,8))

print(data.classes, data.c, len(data.train_ds), len(data.valid_ds), len(data.test_ds))

# can use resnet50 for more layers
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(3)
learn.save('stage-1')

learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
learn.save('stage-2')

# Load ASL Trained CNN
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Example prediction
preds = []
for i in range(0,30):
    # data.test_ds.x[i], can iterate through this array for test images
    p = learn.predict(data.test_ds.x[i])
    preds.append(str(p[0]))
    
print(preds)

interp.plot_top_losses(9, figsize=(10,10))


# Testing
learn.export()
# This will create a file named 'export.pkl' in the directory 
# where we were working that contains everything we need to deploy 
# our model (the model, the weights but also some metadata like the classes or the transforms/normalization used).
defaults.device = torch.device('cpu')
# Open test image
img = open_image(path/'test'/'N'/'IMG_20190225_224044.jpg')
learn = load_learner(path)
pred_class,pred_idx,outputs = learn.predict(img)
# Print the predition
print(pred_class)