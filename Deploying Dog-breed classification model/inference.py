import tensorflow as tf
import numpy as np
import json 
import requests
import config

# convert the unique_breeds list to numpy array
unique_breeds = np.array(config.UNIQUE_BREEDS)

# Turn prediction probabilities into their respective label(easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label
  """
  return unique_breeds[np.argmax(prediction_probabilities)]


def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path,target_size=(config.IMG_SIZE,config.IMG_SIZE)
    )

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image,axis=0)

    data = json.dumps({
        'instances':image.tolist()
    })

    response = requests.post(config.MODEL_URI,data=data.encode())

    result = json.loads(response.text)
    
    print(len(result['predictions'][0]))
    # np.squeeze to remove the extra dimension
    pred_list = np.squeeze(result['predictions'][0])
    #print(get_pred_label(pred_list))

    predicted_label = get_pred_label(pred_list)
    print("Dog breed of: ",predicted_label)
    return predicted_label
