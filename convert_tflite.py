from MnasNet_models import Build_MnasNet
import numpy as np
import tensorflow as tf
import argparse

def model_compare(tf_model, tflite_model):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    # Test the TensorFlow model on random input data.
    tf_results = tf_model(tf.constant(input_data))

    # Compare the result.
    for tf_result, tflite_result in zip(tf_results, tflite_results):
        np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert TF2 model to TFlite.')
    parser.add_argument('-t', '--type', choices=['savedmodel', 'tf_keras'],
                     help='type of input model', required=True)
    parser.add_argument('-i', '--model-path', help='path to the model (depending on the specified type)', required=True)
    parser.add_argument('-o', '--output-path', help='path to output tflite file', required=True)
    args = parser.parse_args()
    
    # Load model
    if args.type == 'savedmodel':
        converter = tf.lite.TFLiteConverter.from_saved_model(args.model_path)
    else:
        model = tf.keras.models.load_model(args.model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    output = args.output_path
    if output.endswith('.tflite'):
        output = output[:-7]
    
    # Convert the model.
    tflite_model = converter.convert()
    open(output + ".tflite", "wb").write(tflite_model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    open(output + "_quantized.tflite", "wb").write(tflite_model)

