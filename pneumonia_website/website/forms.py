from django import forms

class PhotoUploadForm(forms.Form):
    image = forms.ImageField()