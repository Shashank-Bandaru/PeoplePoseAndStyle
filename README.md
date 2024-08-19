# PeoplePoseAndStyle
## Overview
This is a user-friendly web portal designed for managing images and videos. Users can register, log in, and upload their media files, which can then be accessed and viewed securely. Users can even check whether the image contains multiple or single person , their pose (standing or sitting) as well change or remove the background of thier uploaded images.

## Key Features
 ### User Registration and Login :

**Registration**: Users can create a new account by providing a username, email, and password along with password's strength.

**Login**: Registered users can log in with their credentials.

 ### Media Upload :

**Image/Video Upload**: Users can upload images and videos in various formats like jpg, jpeg, png & mp4.

**File Validation**: The system ensures that only allowed file types are uploaded.

 ### Media Display :

**Viewing Media**: Uploaded images and videos can be viewed within the portal.

**Playback Feature**: For videos, a play button allows users to start the video playback directly on the platform.

### Profile :
1. **View User Profile**: Users can see their Username and Email ID along with an default profile pic.
2. **Profile Update**: Users can update their Username, Email ID and their profile pic.

### Individual and Pose prediction :
Users can use the **"predict"** option for the images uploaded allowing them to know whether the image contains : **"Single Person"** , **"Multiple People"** or **"No People"** and in case of a single person being present the model predicts whether the individual is either **"Sitting"** or **"Standing"**.

### Remove Background and Custom Background : 
Users can use the **"remove background"** feature to perform background removal upon the uploaded images. Users can also change the image background by using the **"custom background"** feature.

## Tech Stack
**Frontend**: HTML, CSS, JavaScript

**Backend**: Python (Django)

**Database**: SQLite

## Necessary Libraries
The following libraries need to be installed to run the code:
- Django: Use the command `pip install django`
- After dowloading the code file run the below code to install necessary packages :
  ```
   cd .\Project
   pip install -r requirements.txt
  ```

## Instructions
1. Download the code files from the repo and unzip the folder.
2. Install the necessary libraries as mentioned above.
3. Please download the models from the below links and place them in the Project/DL_models folder :
   <br/> **people_model** : [people_model.keras](https://drive.google.com/file/d/1NzT6yCtdf96XFo48crLWECmJJwGs7c-O/view?usp=sharing)
   <br/> **pose_model** : [pose_model.keras](https://drive.google.com/file/d/12lKIrVQSR1h3KKnVeh7aPxuYE1vH7bfi/view?usp=sharing)
    <br/> Note : Incase you are facing issues when loading the pose model please use this alternative pose_model (it has lesser accuracy) : [pose_model_2.keras](https://drive.google.com/file/d/1QfpzGn05QXkCgCUhgW2zZ4U9vVamuhQL/view?usp=sharing)
5. Make the required mirations using the command `python manage.py makemigrations` after that run the command `python manage.py migrate`.
6. Run the Django server using the command `python manage.py runserver`.
7. Access the web application through your web browser by visiting `http://localhost:8000`.

## Demo
1. The demo video for user account creation ,login and file upload :
   [Django_frontend_demo.mp4](https://drive.google.com/file/d/1uKc067e1OgBc7S6Wd7Dnio3QsImuSlMl/view?usp=sharing)
2. The demo video for individual and pose prediction : [People_pose_predict.mp4](https://drive.google.com/file/d/1kF-2t5lmZ2v5n0QQMXSfEW54ccZAjxH2/view?usp=sharing)
3. The demo video for remove background and custom background functionalities : [Style_Background.mp4](https://drive.google.com/file/d/1cDZmQCdV9vuX6H9xbXZx_4bnD3bGXm9F/view?usp=sharing)
