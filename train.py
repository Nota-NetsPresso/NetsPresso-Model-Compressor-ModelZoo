import os
import math
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import cv2
from albumentations import Compose, PadIfNeeded, RandomCrop, HorizontalFlip, Normalize

class Dataloader(Sequence):
    def __init__(self, x_set, y_set, batch_size, mode="train", shuffle=True):
        self.x, self.y = x_set, to_categorical(y_set)
        self.batch_size = batch_size
        self.mode = mode
        self.augment = self._set_augmentation()
        self.shuffle=shuffle
        self.on_epoch_end()
    
    def _set_augmentation(self):
        if self.mode == "train":
            img_preprocessor = Compose([
                PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                RandomCrop(32, 32),
                HorizontalFlip(p=0.5),
                Normalize (mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), max_pixel_value=255.0)
            ])
        elif self.mode == "test":
            img_preprocessor = Compose([
                Normalize (mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), max_pixel_value=255.0)
            ])
        return img_preprocessor

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

class Trainer():
    def __init__(self, train_config):
        self.model = tf.keras.models.load_model(train_config.model_path)
        os.makedirs(train_config.save_path, exist_ok=True)
        self.save_path = os.path.join(train_config.save_path, "trained_model.h5")
        self.learning_rate = train_config.learning_rate
        self.epochs = train_config.epochs

    def _create_callbacks(self):
        reduce_lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", mode="max", min_delta=0.0001)
        checkpoint = ModelCheckpoint(
                        self.save_path,
                        monitor="val_categorical_accuracy",
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=False,
                        mode="max",
                        save_freq="epoch",
                    )
        return [reduce_lr, checkpoint]

    def do_train(self, train_data, test_data):
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics="CategoricalAccuracy",
        )
        self.model.fit(
            x=train_data,
            validation_data=test_data,
            epochs=self.epochs,
            callbacks=self._create_callbacks(),
        )

        model = tf.keras.models.load_model(self.save_path)

        return model

def main(args):
    # Step 1. Load Dataset
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
    train_loader = Dataloader(train_x, train_y, args.batch_size, "train")
    test_loader = Dataloader(test_x, test_y, args.batch_size, "test")

    # Step 2. Train Model
    trainer = Trainer(train_config=args)
    trained_model = trainer.do_train(train_loader, test_loader)

    # Step 3. Evaluate Model
    trained_model_acc = trained_model.evaluate(x=test_loader)[1] * 100
    print(f"Acc(%) : {round(trained_model_acc, 2)}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, required=True, default="models/cifar100/vgg19.h5", 
        help="input model path, default=%(default)s")
    argparser.add_argument("--save_path", type=str, required=True, default="./", 
        help="saved model path, default=%(default)s")
    argparser.add_argument("--learning_rate", type=float, required=False, default=0.01, 
        help="Initial learning rate, default=%(default)s")
    argparser.add_argument("--batch_size", type=int, required=False, default=128,
        help="Batch size for train, default=%(default)s")
    argparser.add_argument("--epochs", type=int, required=False, default=100, 
        help = "Total training epochs, default=%(default)s")
    args = argparser.parse_args()

    main(args)