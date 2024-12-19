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

            # Run the image through the model for prediction
            result = runImageTesting(image)
            print(result)

            # Create a response string with the file path and result
            response_str = f"Image analyzed. Result: {result} % of pneumonia"

            # Return the result as a string in the HTTP response
            return HttpResponse(response_str)

    else:
        form = PhotoUploadForm()

    return render(request, 'upload_photo.html', {'form': form})