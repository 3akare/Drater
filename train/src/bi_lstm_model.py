from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore

def build_bilstm_classifier(input_shape, hidden_size, num_layers, output_dim, dropout_rate):
    """
    Builds a Bidirectional LSTM classifier model using the Keras Functional API.
    """
    model_input = keras.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(model_input)
    for i in range(num_layers):
        return_sequences = True if i < num_layers - 1 else False
        x = layers.Bidirectional(
            layers.LSTM(hidden_size, return_sequences=return_sequences)
        )(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(output_dim, activation='softmax')(x)
    model = models.Model(inputs=model_input, outputs=output, name='BiLSTM_GestureClassifier')
    return model