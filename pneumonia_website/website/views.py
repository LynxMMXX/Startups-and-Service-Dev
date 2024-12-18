import os
from django.shortcuts import render
from django.http import HttpResponse
from .forms import PhotoUploadForm
from PIL import Image
from .Pneumonia_Model_script import runImageTesting
import io
from django.conf import settings

def testing(request):
    if request.method == 'POST' and request.FILES.get('image'):
        form = PhotoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded image from the request
            image_file = request.FILES['image']

            # Open the image using PIL (Python Imaging Library)
            image = Image.open(image_file)

            save_directory = os.path.join(settings.BASE_DIR, 'website/test_images_Pnmonia/uploaded_images')

            # Make sure the 'uploaded_images' directory exists, create it if not
            os.makedirs(save_directory, exist_ok=True)

            # Empty the directory before saving the new image
            for filename in os.listdir(save_directory):
                file_path = os.path.join(save_directory, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)  # Delete file
                    # If you want to also remove subdirectories (if any), you can add:
                    # elif os.path.isdir(file_path):
                    #     os.rmdir(file_path)  # Remove subdirectory (if it's empty)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

            # Define the file name for the uploaded image
            image_name = 'testImage.png'  # You can also use image_file.name to keep the original file name

            # Full path to save the image
            save_path = os.path.join(save_directory, image_name)

            # Save the image to the 'uploaded_images' directory
            image.save(save_path)

            # Optionally, run some test on the image (like pneumonia model)
            result = runImageTesting()[1:-2]
            print(result)

            # Create a response string with the file path and result
            response_str = f"Image analyzed. Result: {result} % of pneumonia"

            # Return the result as a string in the HTTP response
            return HttpResponse(response_str)

    else:
        form = PhotoUploadForm()

    return render(request, 'upload_photo.html', {'form': form})
