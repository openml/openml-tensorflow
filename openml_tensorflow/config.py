# Config file to define all hyperparameters
from tensorflow.keras.preprocessing.image import ImageDataGenerator

epoch = 10
batch_size = 32
datagen= None
step_per_epoch = 100
target_size = (128,128)
x_col  = None
y_col = None # TODO: Remove? This is not used if a task is defined.

perform_validation = False 
validation_split = 0.1 # the percentage of data set aside for the validation set
validation_steps = 1
datagen_valid = ImageDataGenerator()
kwargs = {}