import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import io
from urllib.request import urlretrieve
import torch
from PIL import Image
from torchvision import transforms


# Custom CSS for background and text
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(
    135deg,
    #f0e500,
    #ffda64,
    #91eae4,
    #91eae4,
    #ffb997,
    #ffb997
  );
  animation: gradientAnimation 15s ease infinite;
  background-size: 600% 600%;
}

@keyframes gradientAnimation {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

.stTitle {
    color: white;
    text-align: center;
}

.stGradientText {
    position: relative;
    display: inline-block;
    padding: 0 8px;
    background: linear-gradient(
        135deg,
        #ff9966,
        #ff5e62,
        #ff2a68,
        #e90072,
        #c7007e,
        #a2008b
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

[data-testid="stHeader"] {
    background-color: black; /* Your custom header bar color */
    color: white; /* Text color for the header */
}

.stImageContainer {
    display: flex;
    justify-content: center;
    align-items: center; /* Center align vertically */
    flex-direction: column; /* Stack items vertically */
}

.stImageContainer img {
    max-width: 80%; /* Adjust the maximum width as needed */
    max-height: 80%; /* Adjust the maximum height as needed */
    object-fit: contain;
}

.stResultText {
    text-align: center;
    color: white;
    margin-top: 10px; /* Add margin for separation */
}
</style>
"""

st.markdown(page_bg_img,unsafe_allow_html=True)

class_names = ['Multiple_People', 'No_People', 'Single_Person']
mapping = ['sitting', 'standing', 'Neither Sitting nor Standing']

def load_model():
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
  model.eval()
  return model

def make_transparent_foreground(pic, mask):
  # split the image into channels
  b, g, r = cv2.split(np.array(pic).astype('uint8'))
  # add an alpha channel with and fill all with transparent pixels (max 255)
  a = np.ones(mask.shape, dtype='uint8') * 255
  # merge the alpha channel back
  alpha_im = cv2.merge([b, g, r, a], 4)
  # create a transparent background
  bg = np.zeros(alpha_im.shape)
  # setup the new mask
  new_mask = np.stack([mask, mask, mask, mask], axis=2)
  # copy only the foreground color pixels from the original image where mask is set
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

  return foreground

def remove_background(model, input_file):
  input_image = input_image = Image.open(input_file).convert("RGB")
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a binary (black and white) mask of the profile foreground
  mask = output_predictions.byte().cpu().numpy()
  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(np.uint8)

  foreground = make_transparent_foreground(input_image ,bin_mask)

  return foreground, bin_mask

def cross_checking(testing_image,model):
    # # Removing the background from the given Image 
    foreground,bin_mask = remove_background(deeplab_model,testing_image)
    output = Image.fromarray(foreground)
    image_bytes = io.BytesIO()
    output.save(image_bytes, format='PNG')
    image_loaded_2 = Image.open(image_bytes).resize((224, 224))
    img = np.array(image_loaded_2)

    # Ensure the image has 3 channels
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    score2 = tf.nn.softmax(model.predict(img[None,:,:]))
    p = np.argmax(score2)
    return [p,score2]


def make_prediction(testing_image,model):
    image_loaded_2 = Image.open(testing_image).resize((224, 224))
    img = np.array(image_loaded_2)

    # Ensure the image has 3 channels
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    score2 = tf.nn.softmax(model.predict(img[None, :, :]))
    predict_index = np.argmax(score2)
    final_predict = predict_index
    final_score = score2
    if(predict_index != 1) :  
        [cross_predict_index, score] = cross_checking(testing_image,model)
        final_predict = cross_predict_index if (
            cross_predict_index == 2 and np.max(score) >= np.max(score2)) else predict_index
        final_score = score if (cross_predict_index == 2 and np.max(
            score) >= np.max(score2)) else score2
    confidence = np.max(final_score) * 100
    print('Predicted class : ', class_names[final_predict])
    print(f"Confidence: {confidence:.2f}%")
    return final_predict

people_model = tf.keras.models.load_model('people_model.keras')
pose_model = tf.keras.models.load_model('best_model10.keras')
deeplab_model = load_model()
st.markdown(
    '<h2 class="stTitle">Pose Estimation: <span class="stGradientText">Standing</span> or <span class="stGradientText">Sitting</span></h1>', unsafe_allow_html=True)

file = st.file_uploader(
    "Please upload image to check the posture", type=["jpg", "png"])


def preprocess_image(image, target_size=(224, 224)):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Get the dimensions of the image
        h, w, _ = img.shape

        # Determine the padding to make the image square
        if h > w:
            pad_width = (h - w) // 2
            padding = ((0, 0), (pad_width, h - w - pad_width), (0, 0))
        else:
            pad_height = (w - h) // 2
            padding = ((pad_height, w - h - pad_height), (0, 0), (0, 0))

        # Pad the image
        img = np.pad(img, padding, mode='constant', constant_values=255)
        assert img.shape[0] == img.shape[1], "Image is not square after padding"

        # Resize the image to the target size
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        assert img.shape[0] == target_size[0] and img.shape[1] == target_size[1], "Image resizing failed"
        img = img / 255.0
        # Expand dimensions to match the model input
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def prediction_pose(image_path,image_bytes2, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    final_score = np.max(predictions)
    print(mapping[predicted_class])
    # print(predictions)
    if(predicted_class!= 2) :  
        [cross_predict_index, score] = cross_checking(image_bytes2,model)
        predicted_class = cross_predict_index if (np.max(score) >= np.max(predictions)) else predicted_class
        final_score = score if (np.max(score) >= np.max(predictions)) else final_score
    confidence = final_score*100
    print(f"Predicted class: {mapping[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    return predictions

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.markdown('<div class="stImageContainer">', unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    
    # Convert PIL Image to BytesIO object for `make_prediction` and `cross_checking`
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    prediction1 = make_prediction(image_bytes, people_model)
    

    # Display result text centered under the image
    if prediction1 != 2:
        if prediction1 == 1:
            st.write("### Predicted Class: No People")
            st.write('##### Note: This image contains no people or in case of only a single person being present the person is not clearly distinguishable from the background. Please upload a clear image of a single person to perform posture detection.')
        else:
            st.write("### Predicted Class: Multiple People")
            st.write('##### Note: This image contains multiple people or in case of only a single person being present there are too many background objects in it. Please upload a clear image of a single person to perform posture detection.')
    else:
        prediction2 = prediction_pose(image, image_bytes,pose_model)
        st.write("### Predicted Class: Single Person Image")
        st.write('##### Note: This image contains a clear view of a single person')
        prediction2 = np.argmax(prediction2)
        if prediction2 == 0:
            st.write("#### Predicted Posture :  Sitting")

        elif prediction2 == 1:
            st.write("#### Predicted Posture :  Standing")
        else:
            st.write("#### This image cannot be classified into either sitting or standing")
        st.write('Note: In case of inaccurate result please try to provide a full body image for obtaining correct prediction')
    st.markdown('</div>', unsafe_allow_html=True)  # Close the result text container
