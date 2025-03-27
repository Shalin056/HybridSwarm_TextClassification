# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding
# from transformers import TFAutoModel, AutoTokenizer

# # 1. LSTM Model
# def build_lstm_model(input_dim, output_dim, lstm_units=64, dropout_rate=0.5):
#     """
#     Build an LSTM-based model for text classification.
#     Args:
#         input_dim (int): Number of input features (e.g., TF-IDF vector length).
#         output_dim (int): Number of classes (e.g., 4 for AG News).
#         lstm_units (int): Number of LSTM units.
#         dropout_rate (float): Dropout rate for regularization.
#     """
#     model = Sequential([
#         tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)),
#         LSTM(lstm_units, return_sequences=False),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # 2. CNN Model
# def build_cnn_model(input_dim, output_dim, filters=64, kernel_size=3, dropout_rate=0.5):
#     """
#     Build a CNN-based model for text classification.
#     Args:
#         input_dim (int): Number of input features (e.g., TF-IDF vector length).
#         output_dim (int): Number of classes.
#         filters (int): Number of filters in Conv1D layer.
#         kernel_size (int): Size of the convolution window.
#         dropout_rate (float): Dropout rate.
#     """
#     model = Sequential([
#         tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
#         Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='valid'),
#         MaxPooling1D(pool_size=2),
#         Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', padding='valid'),
#         GlobalMaxPooling1D(),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # 3. BERT Model
# def build_bert_model(model_name='bert-base-uncased', output_dim=4, max_length=128, trainable=True, learning_rate=2e-5):
#     """
#     Build a BERT-based model for text classification.
#     Args:
#         model_name (str): Pretrained BERT model name from Hugging Face.
#         output_dim (int): Number of classes.
#         max_length (int): Maximum sequence length for tokenization.
#         trainable (bool): Whether to fine-tune BERT layers.
#     """
#     # Load BERT backbone
#     bert = TFAutoModel.from_pretrained(model_name)
#     bert.trainable = trainable  # Freeze or fine-tune BERT

#     # Define input layers
#     input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
#     attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')

#     # Get BERT outputs
#     bert_output = bert(input_ids, attention_mask=attention_mask)[0]  # [batch_size, max_length, hidden_size]
#     cls_token = bert_output[:, 0, :]  # Take [CLS] token for classification

#     # Add classification head
#     x = Dropout(0.3)(cls_token)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     output = Dense(output_dim, activation='softmax')(x)

#     # Build and compile model
#     model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# if __name__ == "__main__":
#     # Test model builds (optional)
#     lstm = build_lstm_model(input_dim=5000, output_dim=4)
#     cnn = build_cnn_model(input_dim=5000, output_dim=4)
#     bert = build_bert_model()
#     print("LSTM Summary:")
#     lstm.summary()
#     print("\nCNN Summary:")
#     cnn.summary()
#     print("\nBERT Summary:")
#     bert.summary()

#v2

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Embedding
# from transformers import TFAutoModel

# # 1. LSTM Model (unchanged)
# def build_lstm_model(input_dim, output_dim, lstm_units=64, dropout_rate=0.5):
#     model = Sequential([
#         tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)),
#         LSTM(lstm_units, return_sequences=False),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # 2. CNN Model (unchanged)
# def build_cnn_model(input_dim, output_dim, filters=64, kernel_size=3, dropout_rate=0.5):
#     model = Sequential([
#         tf.keras.layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
#         Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='valid'),
#         MaxPooling1D(pool_size=2),
#         Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', padding='valid'),
#         GlobalMaxPooling1D(),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # 3. BERT Model (updated)
# def build_bert_model(model_name='bert-base-uncased', output_dim=4, max_length=128, trainable=True):
#     """
#     Build a BERT-based model for text classification.
#     Args:
#         model_name (str): Pretrained BERT model name.
#         output_dim (int): Number of classes.
#         max_length (int): Maximum sequence length (for reference, not used in model definition).
#         trainable (bool): Whether to fine-tune BERT layers.
#     """
#     # Load BERT backbone
#     bert = TFAutoModel.from_pretrained(model_name)
#     bert.trainable = trainable  # Freeze or fine-tune BERT

#     # Define a custom Keras model
#     class BertClassifier(tf.keras.Model):
#         def __init__(self, bert, output_dim):
#             super(BertClassifier, self).__init__()
#             self.bert = bert
#             self.dropout = Dropout(0.3)
#             self.dense1 = Dense(128, activation='relu')
#             self.dropout2 = Dropout(0.3)
#             self.dense2 = Dense(output_dim, activation='softmax')

#         def call(self, inputs, training=False):
#             input_ids = inputs['input_ids']
#             attention_mask = inputs['attention_mask']
#             bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]  # [batch_size, max_length, hidden_size]
#             cls_token = bert_output[:, 0, :]  # Take [CLS] token
#             x = self.dropout(cls_token, training=training)
#             x = self.dense1(x)
#             x = self.dropout2(x, training=training)
#             return self.dense2(x)

#     # Instantiate and compile
#     model = BertClassifier(bert, output_dim)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# if __name__ == "__main__":
#     lstm = build_lstm_model(input_dim=5000, output_dim=4)
#     cnn = build_cnn_model(input_dim=5000, output_dim=4)
#     bert = build_bert_model()
#     print("LSTM Summary:")
#     lstm.summary()
#     print("\nCNN Summary:")
#     cnn.summary()
#     # Note: BERT summary won't work without building with inputs


#v3

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
# from transformers import TFAutoModel

# # 1. LSTM Model (Fixed Reshaping for TF-IDF)
# def build_lstm_model(input_dim, output_dim, lstm_units=64, dropout_rate=0.5):
#     model = Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),  # Proper input layer
#         tf.keras.layers.Reshape((1, input_dim)),    # Keep for compatibility with current usage
#         LSTM(lstm_units, return_sequences=False),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # 2. CNN Model (Adjusted Parameters)
# def build_cnn_model(input_dim, output_dim, filters=128, kernel_size=5, dropout_rate=0.5):
#     model = Sequential([
#         tf.keras.layers.Input(shape=(input_dim,)),  # Proper input layer
#         tf.keras.layers.Reshape((input_dim, 1)),    # Keep for compatibility
#         Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='valid'),
#         MaxPooling1D(pool_size=2),
#         Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', padding='valid'),
#         GlobalMaxPooling1D(),
#         Dropout(dropout_rate),
#         Dense(32, activation='relu'),
#         Dropout(dropout_rate),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # 3. BERT Classifier (Standalone Class)
# class BertClassifier(tf.keras.Model):
#     def __init__(self, bert, output_dim):
#         super(BertClassifier, self).__init__()
#         self.bert = bert
#         self.dropout = Dropout(0.3)
#         self.dense1 = Dense(128, activation='relu')
#         self.dropout2 = Dropout(0.3)
#         self.dense2 = Dense(output_dim, activation='softmax')

#     def call(self, inputs, training=False):
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']
#         bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
#         cls_token = bert_output[:, 0, :]
#         x = self.dropout(cls_token, training=training)
#         x = self.dense1(x)
#         x = self.dropout2(x, training=training)
#         return self.dense2(x)

# # 4. BERT Model (Added Learning Rate Parameter)
# def build_bert_model(model_name='bert-base-uncased', output_dim=4, max_length=128, trainable=True, learning_rate=2e-5):
#     bert = TFAutoModel.from_pretrained(model_name)
#     bert.trainable = trainable
#     model = BertClassifier(bert, output_dim)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model

# if __name__ == "__main__":
#     lstm = build_lstm_model(input_dim=5000, output_dim=4)
#     cnn = build_cnn_model(input_dim=5000, output_dim=4)
#     bert = build_bert_model()
#     print("LSTM Summary:")
#     lstm.summary()
#     print("\nCNN Summary:")
#     cnn.summary()

#v4

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from transformers import TFAutoModel
from transformers import TFBertModel

class BertClassifier(tf.keras.Model):
    def __init__(self, trainable=True, learning_rate=2e-5, **kwargs):
        super(BertClassifier, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased', trainable=trainable)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(4, activation='softmax')
        self.learning_rate = learning_rate
        self.trainable = trainable  # Store trainable as an attribute
        
        # Compile the model during initialization
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def call(self, inputs, training=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        bert_output = self.bert(input_ids, attention_mask=attention_mask, training=training)
        pooled_output = bert_output[1]  # Use [CLS] token output
        dropped = self.dropout(pooled_output, training=training)
        return self.classifier(dropped)

    def get_config(self):
        config = super(BertClassifier, self).get_config()
        config.update({
            'trainable': self.trainable,
            'learning_rate': self.learning_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_lstm_model(input_dim, output_dim, lstm_units=64, dropout_rate=0.2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_dim, output_dim, filters=64, kernel_size=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bert_model(trainable=True, learning_rate=2e-5):
    return BertClassifier(trainable=trainable, learning_rate=learning_rate)