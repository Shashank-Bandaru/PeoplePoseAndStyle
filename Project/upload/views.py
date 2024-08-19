import numpy as np
import io
import cv2
import tensorflow as tf
from PIL import Image, ImageOps
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import CreateUserForm, UserUpdateForm, ProfileUpdateForm
from django.core.exceptions import ObjectDoesNotExist
from .models import MediaFile, Profile
from .decorators import unauthenticated_user, allowed_users
import os
from django.conf import settings
import base64
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import io
from django.core.files.storage import FileSystemStorage


@login_required
def profile(request):
    try:
        profile = request.user.profile
    except ObjectDoesNotExist:
        Profile.objects.create(user=request.user)
        profile = request.user.profile

    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(
            request.POST, request.FILES, instance=request.user.profile)
        print(request.FILES)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, 'Your account has been updated')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form
    }
    return render(request, 'profile.html', context)


@unauthenticated_user
def login_process(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            try:
                profile = user.profile
            except ObjectDoesNotExist:
                Profile.objects.create(user=user)

            login(request, user)
            return redirect('upload')
        else:
            messages.error(request, 'Username or password is incorrect')

    return render(request, 'login.html')


@login_required(login_url='loginPage')
def upload(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        # Save file to media directory
        media_file = MediaFile.objects.create(user=request.user, file=file)
        messages.success(request, 'File uploaded successfully')
        return redirect('upload')

    media_files = MediaFile.objects.filter(user=request.user)
    context = {'user': request.user, 'media_files': media_files}
    return render(request, 'upload.html', context)


def change_slashes(image_path):
    return image_path.replace('\\', '/')


def predictt(request):
    if request.method == 'POST':
        image = request.POST.get('image_url')
        print('returned image:')
        print(image)
        image_relative_path = image
        image_absolute_path = os.path.join(
            settings.BASE_DIR, image_relative_path.lstrip('/'))

        prediction_result = [0, 0]
        print(image_absolute_path)
        image_path = change_slashes(image_absolute_path)
        prediction_result = prediction_of_image(image_path)
        print(image_path)
        context = {'image_src': image, 'prediction_results': prediction_result}
        return render(request, 'results.html', context)
    return redirect('upload')


def custombg(request):
    if request.method == 'POST':
        original_image = request.POST.get('image_url')
        rembg_image = request.POST.get('rembg_url')
        processed_image = None
        action = request.POST.get('action')

        if action == 'action1':
            image = original_image
            print('returned image:')
            print(image)
            image_absolute_path = os.path.join(
                settings.BASE_DIR, image.lstrip('/'))
            image = change_slashes(image_absolute_path)
            foreground, bin_mask = remove_background(deeplab_model, image)
            white_background_path = "media/white_background_output.png"
            white_saving_path = os.path.join(
                settings.MEDIA_ROOT, white_background_path.lstrip('/'))
            white_saving_path = change_slashes(white_saving_path)
            print(white_saving_path)
            make_white_background_image(bin_mask, white_saving_path)
            white_backgrounded_image = custom_background(
                white_saving_path, foreground)
            white_background_path = "/media/media/white_background_output.png"
            white_backgrounded_image.save(white_saving_path)
            processed_image = white_background_path
        elif action == 'action2':
            upload = request.FILES.get('custom_img')
            print("type")
            print(type(upload))
            if upload:
                if upload.size > 2 * 1024 * 1024:
                    return HttpResponse('File size exceeds 2MB.', status=400)
                if upload.content_type not in ['image/jpeg', 'image/png']:
                    return HttpResponse('Invalid file type. Only JPEG and PNG are allowed.', status=400)

                # apply custom background
                fs = FileSystemStorage()
                # Save the uploaded file
                filename = fs.save("custom_img.jpg", upload)
                print("uploaded file name")
                print(upload.name)
                # Get the file's URL
                uploaded_file_url = fs.url(filename)
                file_url = "media/"
                file_url = uploaded_file_url
                print("uploaded custom background file url:")
                print(uploaded_file_url)

                image = original_image
                print('returned image:')
                print(image)
                image_absolute_path = os.path.join(
                    settings.BASE_DIR, image.lstrip('/'))
                image = change_slashes(image_absolute_path)

                foreground, bin_mask = remove_background(deeplab_model, image)

                uploaded_file_url = os.path.join(
                    settings.BASE_DIR, uploaded_file_url.lstrip('/'))
                uploaded_file_url = change_slashes(uploaded_file_url)

                print("final url of uploaded file")
                print(uploaded_file_url)

                final_image = custom_background(uploaded_file_url, foreground)
                if final_image.mode == 'RGBA':
                    final_image = final_image.convert('RGB')

                final_image.save(uploaded_file_url)
                processed_image = file_url
                print("final_url")
                print(file_url)
            else:
                return HttpResponse('No file uploaded.', status=400)

        context = {'image_src': original_image,
                   'processed_image': processed_image, 'rembg_image': rembg_image}
        return render(request, 'rembg.html', context)


def rembg(request):
    processed_image = None
    if request.method == 'POST':
        image = request.POST.get('image_url')
        image_path = request.POST.get('image_url')
        print('returned image:')
        print(image)
        image_absolute_path = os.path.join(
            settings.BASE_DIR, image.lstrip('/'))
        image = change_slashes(image_absolute_path)

        print(image)
        image_relative_saving_path = "media/removed_background.png"
        image_saving_path = os.path.join(
            settings.MEDIA_ROOT, image_relative_saving_path.lstrip('/'))
        image_saving_path = change_slashes(image_saving_path)

        print(image_saving_path)
        # foreground = np.random.rand(100, 100, 4) * 255
        foreground, bin_mask = remove_background(deeplab_model, image)
        image_array = foreground.astype(np.uint8)

        # Convert the NumPy array to an image
        rembg_image = Image.fromarray(image_array, 'RGBA')
        rembg_image.save(image_saving_path)
        image_relative_saving_path = "/media/media/removed_background.png"

        context = {'image_src': image_path,
                   'rembg_image': image_relative_saving_path, 'processed_image': processed_image}
        return render(request, 'rembg.html', context)
    return redirect('results')


def registerPage(request):
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(
                request, 'Account is successfully created for ' + user)
            return redirect('loginPage')
    else:
        form = CreateUserForm()

    context = {'form': form}
    return render(request, 'register.html', context)


@login_required
def logoutuser(request):
    logout(request)
    return redirect('loginPage')


def prediction_of_image(image_path):
    image = Image.open(image_path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)

    prediction1 = make_prediction(image_bytes, people_model)
    prediction2 = prediction(image, pose_model)
    return [prediction1, prediction2]


class_names = ['Multiple_People', 'No_People', 'Single_Person']
mapping = ['sitting', 'standing', 'no_people']


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


people_model = tf.keras.models.load_model('./DL_models/people_model.keras')
pose_model = tf.keras.models.load_model('./DL_models/pose_model.keras')


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

        return None


def prediction(image_path, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = 100 * np.max(predictions)
    # print(predictions)
    print(f"Predicted class: {mapping[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")
    prediction2 = predicted_class
    return prediction2


def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model


def make_transparent_foreground(pic, mask):
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    a = np.ones(mask.shape, dtype='uint8') * 255
    alpha_im = cv2.merge([b, g, r, a], 4)
    bg = np.zeros(alpha_im.shape)
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
    return foreground


def remove_background(model, input_file):
    input_image = Image.open(input_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image, bin_mask)

    return foreground, bin_mask


deeplab_model = load_model()


def custom_background(background_file, foreground):
    final_foreground = Image.fromarray(foreground)
    background = Image.open(background_file)

    # Calculate scale to fit foreground in background
    fg_width, fg_height = final_foreground.size
    bg_width, bg_height = background.size

    scale = min(bg_width / fg_width, bg_height / fg_height)
    new_fg_width = int(fg_width * scale)
    new_fg_height = int(fg_height * scale)

    final_foreground = final_foreground.resize(
        (new_fg_width, new_fg_height), Image.Resampling.LANCZOS)

    # Center the resized foreground on the background
    x = (bg_width - new_fg_width) // 2
    y = (bg_height - new_fg_height) // 2

    background.paste(final_foreground, (x, y), final_foreground)
    return background


def make_white_background_image(bitmask, path):
    array = bitmask

    # Create a 3-channel image (RGB mode)
    height, width = array.shape
    image = Image.new('RGB', (width, height))

    # Iterate through the array and set pixel values
    for y in range(height):
        for x in range(width):
            pixel_value = (0, 0, 0) if array[y, x] == 1 else (255, 255, 255)
            image.putpixel((x, y), pixel_value)

    # Save the image
    image.save(path)
