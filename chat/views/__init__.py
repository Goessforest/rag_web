# chat/views.py
import os
# import openai
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import HttpResponseBadRequest
import logging
from .chat import chat_functionality

from ..rag.vector_retriever import VectorDBRetriever
from ..rag import RAG_defaults

from .request_handler import Chat_home_model


from django.contrib.auth.decorators import login_required


@login_required(login_url='/login/')  # Redirects to login page if not authenticated
@csrf_exempt
def chat_home(request):
    if request.user.is_authenticated:
        model = Chat_home_model(request)

        return render(request, 'chat/chat_home.html', model.to_dict())
    
    else: # Redirect to login page if not authenticated
        logging.error(f"User {request.user} not authenticated")
        return render(request, 'chat/login.html')
