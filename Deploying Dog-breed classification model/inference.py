import tensorflow as tf
import numpy as np
import json
import requests

# image size that the trained model excepts
IMG_SIZE =224

# model served on tensorflow serving using docker
MODEL_URI ='http://localhost:8501/v1/models/dogbreed:predict'


# unique breeds(labels) of dogs that the model was trained on
unique_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier',
 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog',
 'black-and-tan_coonhound', 'blenheim_spaniel' ,'bloodhound', 'bluetick',
 'border_collie', 'border_terrier', 'borzoi', 'boston_bull',
 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard',
 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel',
 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
 'doberman', 'english_foxhound', 'english_setter', 'english_springer',
 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog',
 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer',
 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees',
 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter',
 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie',
 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever',
 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
 'mexican_hairless' ,'miniature_pinscher', 'miniature_poodle',
 'miniature_schnauzer', 'newfoundland' ,'norfolk_terrier',
 'norwegian_elkhound' ,'norwich_terrier', 'old_english_sheepdog',
 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug',
 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki',
 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky',
 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff',
 'tibetan_terrier' ,'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound',
 'weimaraner' ,'welsh_springer_spaniel', 'west_highland_white_terrier',
 'whippet' ,'wire-haired_fox_terrier' ,'yorkshire_terrier']


# convert the unique_breeds list to numpy array
unique_breeds = np.array(unique_breeds)

# Turn prediction probabilities into their respective label(easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label
  """
  return unique_breeds[np.argmax(prediction_probabilities)]


def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path,target_size=(IMG_SIZE,IMG_SIZE)
    )

    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image,axis=0)

    data = json.dumps({
        'instances':image.tolist()
    })

    response = requests.post(MODEL_URI,data=data.encode())

    result = json.loads(response.text)
    
    print(len(result['predictions'][0]))
    # np.squeeze to remove the extra dimension
    pred_list = np.squeeze(result['predictions'][0])
    #print(get_pred_label(pred_list))

    predicted_label = get_pred_label(pred_list)
    print("Dog breed of: ",predicted_label)
    return predicted_label
