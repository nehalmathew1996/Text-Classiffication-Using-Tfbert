import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from functions import generate_training_data,SentimentDatasetMapFunction
from transformers import TFBertModel

# Reading data
df = pd.read_csv('training.csv')

# model 
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

X_input_ids = np.zeros((len(df), 256))
X_attn_masks = np.zeros((len(df), 256))

# Tokenization getting mask and padding
X_input_ids, X_attn_masks = generate_training_data(df, X_input_ids, X_attn_masks, tokenizer)

# Output label file
labels = np.zeros((len(df), 6))

# one-hot encoded target tensor
labels[np.arange(len(df)), df['label'].values] = 1 

# creating a data pipeline using tensorflow dataset utility, creates batches of data for easy loading...
dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))

# converting to required format for tensorflow dataset 
dataset = dataset.map(SentimentDatasetMapFunction) 

# batch size, drop any left out tensor
dataset = dataset.shuffle(10000).batch(64, drop_remainder=True) 

# for each 16 batch of data we will have len(df)//16 samples, take 80% of that for train.
p = 0.8
train_size = int((len(df))*p) 

# Train test split
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


# Model Building
model = TFBertModel.from_pretrained('bert-base-cased') # bert base model with pretrained weights

# defining 2 input layers for input_ids and attn_masks
input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1] # 0 -> activation layer (3D), 1 -> pooled output layer (2D)
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(6, activation='softmax', name='output_layer')(intermediate_layer) # softmax -> calcs probs of classes

sentiment_model = tf.keras.Model(inputs=[input_ids, attn_masks], outputs=output_layer)
# sentiment_model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

sentiment_model.compile(optimizer=optim, loss=loss_func, metrics=[acc])

hist = sentiment_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# Change file name
sentiment_model.save('')



