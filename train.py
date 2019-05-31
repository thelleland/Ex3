# The imported generators expect to find training data in data/train
# and validation data in data/validation
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
import keras
import os
from os import makedirs
from os.path import exists, join
from create_model import create_base_network, in_dim, tripletize, std_triplet_loss
from generators import triplet_generator
import testing as T
import image_to_numpy
import config as C
import numpy as np
# My imports
from keras.callbacks import ModelCheckpoint
import time
import json

last = C.last




def save_name(i):
    return ('models/epoch_'+str(i)+'.model')

def log(s):
    with open(C.logfile, 'a') as f:
        print(s, file=f)

# Use log to file
logger = CSVLogger(C.logfile, append=True, separator='\t')


def train_base_model(model, optimizer_type='adam', loss_type='categorical_crossentropy', metrics_type='accuracy'):
    
    x = model.output
    predictions = Dense(40, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    model.compile(optimizer=SGD(lr=0.1, momentum=0.9), loss=loss_type, metrics=[metrics_type])

    
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        C.base_train,
        target_size=(299,299),
        color_mode='rgb',
        class_mode='categorical',
        batch_size= C.batch_size)

    validation_generator = val_datagen.flow_from_directory(
        C.val_dir,

        target_size=(299,299),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=C.batch_size,
        seed=223)
    
    if not os.path.exists("./base_model_logs"):
        makedirs("./base_model_logs")



    logger_base = CSVLogger("./base_model_logs/train_base_.log")
    checkpoint_base = ModelCheckpoint("./base_model_logs/weights.h5",
                                      monitor='val_loss',
                                      verbose=1,save_best_only=False,mode='auto',
                                      period=1)
    model.fit_generator(
        train_generator,
        steps_per_epoch= 70,
        epochs = 5,
        callbacks=[logger_base, checkpoint_base],
        validation_data=validation_generator,
        validation_steps=10)

    # Fine tuning 
    for layer in model.layers[300:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9),loss='categorical_crossentropy',metrics=[metrics_type])

    logger_fine = CSVLogger("./base_model_logs/train_fine_{}.log".format(time.time()))
    checkpoint_fine = ModelCheckpoint("./base_model_logs/weights_fine.h5",
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=False,
                                     mode='auto',
                                     period=1)
    print("Fine tuning base_model")
    model.fit_generator(
        train_generator,
        steps_per_epoch=30,
        epochs=5,
        callbacks=[logger_fine,checkpoint_fine],
        validation_data=validation_generator,
        validation_steps=10)
              
    model.layers.pop()
    for layer in model.layers[300:]:
        layer.trainable = False

def train_step():
    history = model.fit_generator(
        triplet_generator(C.batch_size, None, C.train_dir), steps_per_epoch=20, epochs=C.iterations,
        callbacks=[logger],
        validation_data=triplet_generator(C.batch_size, None, C.val_dir), validation_steps=10)
    return history

if last==0:
    log('Creating base network from scratch.')
    if not os.path.exists('models'):
        os.makedirs('models')
    base_model = create_base_network(in_dim)
else:
    #log('Loading model:'+save_name(last))
    #base_model = load_model(save_name(last))
    print()
    


#train_base_model(base_model)
#base_model = load_model("./base_model_logs/weights_fine.h5")
#base_model.layers.pop()
#base_model.summary()

# for layer in base_model.layers:
#     layer.trainable = True



model = tripletize(base_model)
#model.summary()


print("\ntripletized")
model.compile(optimizer=SGD(lr=C.learn_rate, momentum=0.9),
              loss=std_triplet_loss(), metrics=['accuracy'])
print("\ncompiled")

def avg(x):
    return sum(x)/len(x)


def append_hist(from_hist, to_hist):
    for key, values in from_hist.items():
        for value in values:
            to_hist[key].append(value)

            
print("\ngetting vectors")
start_time = time.time()
vs = T.get_vectors(base_model, C.val_dir)
stop_time = time.time()
print("Time to get vectors: ",str(stop_time-start_time))
cents = {}
print("\ngetting centroids")
for v in vs:
    cents[v] = T.centroid(vs[v])
print("\ntraining")

if not exists("./plots"):
    os.makedirs("./plots")

    
T.PCA_plot(base_model, vs, C.val_dir, last, "./plots")
my_history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

for i in range(last+1, last+11):
            
    log('Starting iteration '+str(i)+'/'+str(last+10)+' lr='+str(C.learn_rate))
    history = train_step()
    append_hist(history.history, my_history)

    T.plot_graphs(my_history, i, "./plots")
    
    C.learn_rate = C.learn_rate * C.lr_decay
    base_model.save(save_name(i))

    print("Getting new vectors:")
    start_time = time.time()
    vs = T.get_vectors(base_model, C.val_dir)
    stop_time = time.time()
    
    T.PCA_plot(base_model, vs, C.val_dir, i, "./plots")
    
    print("Time getting vectros: ", str(stop_time - start_time))
    c = T.count_nearest_centroid(vs)
    log('Summarizing '+str(i))
    with open('summarize.'+str(i)+'.log', 'w') as sumfile:
        T.summarize(vs, outfile=sumfile)
    with open('clusters.'+str(i)+'.log', 'w') as cfile:
        T.confusion_counts(c, outfile=cfile)
    c_tmp = {}
    r_tmp = {}
    for v in vs:
        c_tmp[v] = T.centroid(vs[v])
        r_tmp[v] = T.radius(c_tmp[v], vs[v])
    c_rad = [round(100*r_tmp[v])/100 for v in vs]
    c_mv = [round(100*T.dist(c_tmp[v],cents[v]))/100 for v in vs]
    log('Centroid radius: '+str(c_rad))
    log('Centroid moved: '+str(c_mv))
    cents = c_tmp

    with open(C.logfile, 'a') as f:
        T.accuracy_counts(c, outfile=f)
    # todo: avg cluster radius, avg cluster distances
    log('Avg centr rad: %.2f move: %.2f' % (avg(c_rad), avg(c_mv)))
