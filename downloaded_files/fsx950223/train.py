from mobilenetv3 import MobilenetV3
import tensorflow as tf
import tensorflow_datasets as tfds

tf.enable_eager_execution()

BATCH_SIZE = 32
USE_TPU=False
NUM_CLASSES=10
EPOCH=20
DATASET='cifar10'
INPUT_SHAPE=(32,32)

mobilenet_v3 = MobilenetV3(INPUT_SHAPE,NUM_CLASSES,'small',alpha=1.0)
cos_lr = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch, _: tf.train.cosine_decay(1e-3, epoch,EPOCH)().numpy(), 1)
logging=tf.keras.callbacks.TensorBoard(log_dir='./logs', write_images=True)
mobilenet_v3.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.sparse_categorical_crossentropy,
                     metrics=["sparse_categorical_accuracy"])
if USE_TPU:
    tpu=tf.contrib.cluster_resolver.TPUClusterResolver()
    strategy=tf.contrib.tpu.TPUDistributionStrategy(tpu)
    mobilenet_v3=tf.contrib.tpu.keras_to_tpu_model(mobilenet_v3,strategy=strategy)
dataset, info = tfds.load(name=DATASET, split=[tfds.Split.TRAIN, tfds.Split.TEST], with_info=True,as_supervised=True,try_gcs=tfds.is_dataset_on_gcs(DATASET))
train_dataset, test_dataset = dataset
train_num = info.splits['train'].num_examples
test_num = info.splits['test'].num_examples

def preprocess_image(image):
    image=tf.image.random_brightness(image,0.1)
    image=tf.image.random_hue(image,0.1)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.resize(image, INPUT_SHAPE)
    return image

train_dataset = train_dataset.map(lambda image,label:(preprocess_image(image),label)).shuffle(10000).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE).repeat()
test_dataset = test_dataset.map(lambda image,label:(tf.image.resize(image,INPUT_SHAPE),label)).batch(BATCH_SIZE).prefetch(
    tf.data.experimental.AUTOTUNE).repeat()
mobilenet_v3.fit(train_dataset, epochs=EPOCH, steps_per_epoch=max(1,train_num//BATCH_SIZE), validation_data=test_dataset,validation_steps=max(1,test_num//BATCH_SIZE),callbacks=[cos_lr])
mobilenet_v3.save_weights('./mobilenetv3_test.h5')
