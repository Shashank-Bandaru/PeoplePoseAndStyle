from django.db import models
from django.contrib.auth.models import User
from PIL import Image

class MediaFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='media_files')
    file = models.FileField(upload_to='media/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    @property
    def is_image(self):
        return self.file.name.lower().endswith(('.jpg', '.jpeg', '.png'))

    @property
    def is_video(self):
        return self.file.name.lower().endswith(('.mp4'))

    def __str__(self):
        return self.file.name

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    image = models.ImageField(default='profile_pics/default.png', upload_to='profile_pics')

    def __str__(self):
        return f'{self.user.username} Profile'
    
    def save(self, *args, **kwargs):  # Modified save method to accept *args and **kwargs
        super().save(*args, **kwargs)

        img = Image.open(self.image.path)

        if img.height > 300 or img.width > 300:
            output_size = (300, 300)
            img.thumbnail(output_size)
            img.save(self.image.path)
