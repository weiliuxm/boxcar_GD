# -*- coding: utf-8 -*-
import _init_paths
# this should be soon to prevent tensorflow initialization with -h parameter
from utils import ensure_dir, parse_args
args = parse_args(["ResNet50", "VGG16", "VGG19", "InceptionV3"])

# other imports
import os
import time
import sys
from keras import backend as K

from boxcars_dataset_liu import BoxCarsDataset
from boxcars_data_generator import BoxCarsDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard


def tanh_loss(y_true, y_pred):
    loss = -0.5*((1-y_true)*K.log(1-y_pred) + (1+y_true)*K.log(1+y_pred))
    return K.mean(loss,axis=-1)

#%% initialize dataset
if args.estimated_3DBB is None:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
else:
    dataset = BoxCarsDataset(load_split="hard", load_atlas=True, 
                             use_estimated_3DBB = True, estimated_3DBB_path = args.estimated_3DBB)

#%% get optional path to load model
model = None
for path in [args.eval, args.resume]:
    if path is not None:
        print("Loading model from %s"%path)
        model = load_model(path)
        break

#%% construct the model as it was not passed as an argument





print("Model name: %s"%(model.name))
if args.estimated_3DBB is not None and "estimated3DBB" not in model.name:
    print("ERROR: using estimated 3DBBs with model trained on original 3DBBs")
    sys.exit(1)
if args.estimated_3DBB is None and "estimated3DBB" in model.name:
    print("ERROR: using model trained on estimated 3DBBs and running on original 3DBBs")
    sys.exit(1)

args.output_final_model_path = os.path.join(args.cache, model.name, "final_model.h5")
args.snapshots_dir = os.path.join(args.cache, model.name, "snapshots")
args.tensorboard_dir = os.path.join(args.cache, model.name, "tensorboard")

#%% training
if args.eval is None:
    print("Training...")
    #%% initialize dataset for training
    dataset.initialize_data("train")
    dataset.initialize_data("validation")
    generator_train = BoxCarsDataGenerator(dataset, "train", args.batch_size, training_mode=True)
    generator_val = BoxCarsDataGenerator(dataset, "validation", args.batch_size, training_mode=False)
    #%% callbacks
    ensure_dir(args.tensorboard_dir)
    ensure_dir(args.snapshots_dir)
    tb_callback = TensorBoard(args.tensorboard_dir, histogram_freq=1, write_graph=False, write_images=False)
#    saver_callback = ModelCheckpoint(os.path.join(args.snapshots_dir, "model_{epoch:03d}_{val_acc:.2f}.h5"), monitor='val_acc', period=1 )
    saver_callback = ModelCheckpoint(os.path.join(args.snapshots_dir, "model_{epoch:03d}_{class_prediction_acc:.2f}.h5"), monitor='val_acc', period=2 )
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    #%% get initial epoch
    initial_epoch = 0
    if args.resume is not None:
        initial_epoch = int(os.path.basename(args.resume).split("_")[1]) + 1



    history = model.fit_generator(generator=generator_train, 
                        samples_per_epoch=generator_train.n,
                        nb_epoch=args.epochs,
                        verbose=1,
                        validation_data=generator_val,
                        nb_val_samples=generator_val.n,
                        callbacks=[tb_callback, saver_callback],
                        initial_epoch = initial_epoch,
                        )
    print(history.history.keys())

    #%% save trained data

    print("Saving the final model to %s"%(args.output_final_model_path))
    ensure_dir(os.path.dirname(args.output_final_model_path))
    model.save(args.output_final_model_path)


#%% evaluate the model 
print("Running evaluation...")
dataset.initialize_data("test")
generator_test = BoxCarsDataGenerator(dataset, "test", args.batch_size, training_mode=False, generate_y=False)
start_time = time.time()
predictions = model.predict_generator(generator_test, generator_test.n)
end_time = time.time()
single_acc, tracks_acc = dataset.evaluate(predictions)
print(" -- Accuracy: %.2f%%"%(single_acc*100))
print(" -- Track accuracy: %.2f%%"%(tracks_acc*100))
print(" -- Image processing time: %.1fms"%((end_time-start_time)/generator_test.n*1000))
