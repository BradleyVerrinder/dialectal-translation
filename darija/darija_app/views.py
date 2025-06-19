from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import CustomUserCreationForm, LoginForm
from .models import User
from django.contrib.auth.hashers import make_password
from django.contrib.auth import authenticate, login as login2, logout as django_logout
from .decorators import custom_login_required
from .darijatoenglish import load_model, translate
from .englishtodarija import load_model as load_model2, translate as translate2
from transformers import BertTokenizer
from transformers import AutoTokenizer
from django.http import JsonResponse
import logging
import json
from .models import Translation
from django.views.decorators.csrf import csrf_exempt

# Load model globally if it's feasible; otherwise, consider lazy loading techniques
model_darija_to_english = load_model('../translation_model_after_training.pth')
model_english_to_darija = load_model2('../english-darija_translation_model_after_training.pth')

# Assuming tokenizers are globally available after model load
darija_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def home(request):
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Request method: %s", request.method)
    logging.debug("AJAX request: %s", request.headers.get('X-Requested-With') == 'XMLHttpRequest')
    darija_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #context = {
    #    'input_text': '',  # Default empty string
    #    'translated_text': '',
    #    'model_type': 'dar_to_eng'  # Default empty string
    #}
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        input_text = request.POST.get('input_text', 'default fallback if missing')
        #model_type = request.POST.get('model_type', 'dar_to_eng')
        data = json.loads(request.body)
        logging.debug("Received data: %s", data)  # Log the data received
        input_text = data.get('input_text', 'default input')
        model_type = data.get('model_type', 'default model type')
        logging.debug("Input text: %s", input_text) 
        logging.debug("Input text: %s", input_text)
        logging.debug("Model type: %s", model_type)
        if model_type == 'eng_to_dar':
            model = model_english_to_darija
            #context['model_type'] = 'eng_to_dar'
            darija_tokenizer= AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
        else:
            model = model_darija_to_english
            #context['model_type'] = 'dar_to_eng'
            darija_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        # Assume the model and tokenizers are set up correctly and accessible here
        if model_type == 'dar_to_eng':
            translated_text = translate(model, input_text, darija_tokenizer, english_tokenizer)
        else:
            translated_text = translate2(model, input_text, english_tokenizer, darija_tokenizer)
        #context['translated_text'] = translated_text
        #context['input_text'] = input_text
        return JsonResponse({'translated_text': translated_text})
    return render(request, 'translate.html')

@custom_login_required
def translate_admin(request):
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Request method: %s", request.method)
    logging.debug("AJAX request: %s", request.headers.get('X-Requested-With') == 'XMLHttpRequest')
    darija_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        input_text = request.POST.get('input_text', 'default fallback if missing')
        #model_type = request.POST.get('model_type', 'dar_to_eng')
        data = json.loads(request.body)
        logging.debug("Received data: %s", data)  # Log the data received
        input_text = data.get('input_text', 'default input')
        model_type = data.get('model_type', 'default model type')
        logging.debug("Input text: %s", input_text) 
        logging.debug("Input text: %s", input_text)
        logging.debug("Model type: %s", model_type)
        if model_type == 'eng_to_dar':
            model = model_english_to_darija
            #context['model_type'] = 'eng_to_dar'
            darija_tokenizer= AutoTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")
        else:
            model = model_darija_to_english
            #context['model_type'] = 'dar_to_eng'
            darija_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        # Assume the model and tokenizers are set up correctly and accessible here
        if model_type == 'dar_to_eng':
            translated_text = translate(model, input_text, darija_tokenizer, english_tokenizer)
        else:
            translated_text = translate2(model, input_text, english_tokenizer, darija_tokenizer)
        #context['translated_text'] = translated_text
        #context['input_text'] = input_text
        return JsonResponse({'translated_text': translated_text})
    favourites = Translation.objects.filter(user=request.user).order_by('-created_at')

    return render(request, 'translate_admin.html', {'favourites': favourites})

@csrf_exempt  # Use this decorator if CSRF validation is causing issues with fetch requests
def add_to_favourites(request):
    if request.method == 'POST':
        source_text = request.POST.get('source_text', None)
        translated_text = request.POST.get('stored_translation', None)
        
        if not source_text or not translated_text:
            return JsonResponse({'status': 'error', 'message': 'Missing data'}, status=400)

        # Assuming that Translation model has fields `source_text` and `translated_text`
        translation = Translation(user=request.user, source_text=source_text, translated_text=translated_text)
        translation.save()
        
        return JsonResponse({'status': 'success', 'message': 'Translation saved successfully'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request'}, status=400)

def delete_favourite(request, id):
    if request.method == 'POST':
        favourite = Translation.objects.get(pk=id, user=request.user)  # Ensure that the logged-in user can only delete their own entries
        favourite.delete()
    return redirect('translate_admin')  # Redirect back to the favourites page

def login(request):
    if request.user.is_authenticated:
        return redirect('translate_admin')
    
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request=request, username=username, password=password)
            if user is not None:
                login2(request, user)
            return redirect('translate_admin')  # Redirect to home page on successful login
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})

def logout(request):
    django_logout(request)
    return redirect('home') 

def registration(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = User(username = form.cleaned_data['username'],
                        password = make_password(form.cleaned_data['password1']),
                        email = form.cleaned_data['email'])
            user.save()
            return redirect('login')
    else:
        form = CustomUserCreationForm()
    return render (request, 'register.html', {'form': form})
