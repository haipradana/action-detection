import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv3D, BatchNormalization, MaxPool3D, ConvLSTM2D, MaxPool2D, 
    LayerNormalization, Dense, Flatten, Dropout, Activation
)


class Encoder(Model):
    """3D CNN Encoder for extracting spatial-temporal features"""
    
    def __init__(self):
        super(Encoder, self).__init__()
        # First 3D Conv block
        self.conv3d_1 = Conv3D(16, kernel_size=(1, 7, 7), padding='same', name='conv3d_1')
        self.batchnorm_1 = BatchNormalization(name='bn_1')
        self.activation_1 = Activation('relu', name='relu_1')
        self.maxpool3d_1 = MaxPool3D(pool_size=(1, 2, 2), name='maxpool3d_1')
        
        # Second 3D Conv block
        self.conv3d_2 = Conv3D(32, kernel_size=(1, 5, 5), padding='same', name='conv3d_2')
        self.batchnorm_2 = BatchNormalization(name='bn_2')
        self.activation_2 = Activation('relu', name='relu_2')
        self.maxpool3d_2 = MaxPool3D(pool_size=(1, 2, 2), name='maxpool3d_2')
        
        # Third 3D Conv block
        self.conv3d_3 = Conv3D(64, kernel_size=(1, 3, 3), padding='same', name='conv3d_3')
        self.batchnorm_3 = BatchNormalization(name='bn_3')
        self.activation_3 = Activation('relu', name='relu_3')
        self.maxpool3d_3 = MaxPool3D(pool_size=(1, 2, 2), name='maxpool3d_3')

    def call(self, inputs, training=None):
        # First block
        x = self.conv3d_1(inputs)
        x = self.batchnorm_1(x, training=training)
        x = self.activation_1(x)
        x = self.maxpool3d_1(x)
        
        # Second block
        x = self.conv3d_2(x)
        x = self.batchnorm_2(x, training=training)
        x = self.activation_2(x)
        x = self.maxpool3d_2(x)
        
        # Third block
        x = self.conv3d_3(x)
        x = self.batchnorm_3(x, training=training)
        x = self.activation_3(x)
        x = self.maxpool3d_3(x)
        
        return x


class MyCL_Model(Model):
    """ConvLSTM model for action recognition"""
    
    def __init__(self, num_classes=6, dropout_rate=0.3):
        super(MyCL_Model, self).__init__()
        self.num_classes = num_classes
        
        # Encoder for spatial feature extraction
        self.encoder = Encoder()
        
        # ConvLSTM for temporal modeling
        self.convlstm_1 = ConvLSTM2D(
            filters=32, 
            kernel_size=(3, 3), 
            strides=(2, 2),
            padding='valid', 
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='convlstm_1'
        )
        
        # Normalization and pooling
        self.layernorm = LayerNormalization(name='layer_norm')
        self.maxpool2d = MaxPool2D(name='maxpool2d')
        
        # Flatten and dense layers
        self.flatten = Flatten(name='flatten')
        self.dropout_1 = Dropout(dropout_rate, name='dropout_1')
        self.dense_1 = Dense(128, activation='relu', name='dense_1')
        self.dropout_2 = Dropout(dropout_rate, name='dropout_2')
        self.dense_2 = Dense(64, activation='relu', name='dense_2')
        
        # Classifier with float32 output for mixed precision
        self.classifier = Dense(num_classes, activation='softmax', name='classifier', dtype='float32')
        
    def call(self, inputs, training=None):
        # Extract spatial features
        x = self.encoder(inputs, training=training)
        
        # Temporal modeling with ConvLSTM
        x = self.convlstm_1(x, training=training)
        
        # Normalization and pooling
        x = self.layernorm(x, training=training)
        x = self.maxpool2d(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.dropout_1(x, training=training)
        x = self.dense_1(x)
        x = self.dropout_2(x, training=training)
        x = self.dense_2(x)
        
        # Classification
        output = self.classifier(x)
        
        return output
    
    def build_model(self, input_shape):
        """Build the model with specified input shape"""
        inputs = tf.keras.Input(shape=input_shape)
        outputs = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def get_model_summary(self, input_shape):
        """Get model summary"""
        model = self.build_model(input_shape)
        return model.summary()


# Alternative simpler model for comparison
class SimpleConvLSTM(Model):
    """Simplified ConvLSTM model for faster training"""
    
    def __init__(self, num_classes=6):
        super(SimpleConvLSTM, self).__init__()
        
        # Single 3D Conv layer
        self.conv3d = Conv3D(16, kernel_size=(1, 7, 7), padding='same', activation='relu')
        self.maxpool3d = MaxPool3D(pool_size=(1, 2, 2))
        
        # ConvLSTM
        self.convlstm = ConvLSTM2D(
            filters=16, 
            kernel_size=(3, 3), 
            padding='valid', 
            return_sequences=False
        )
        
        # Classification layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv3d(inputs)
        x = self.maxpool3d(x)
        x = self.convlstm(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)