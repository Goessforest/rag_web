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

from .files import handle_files, list_pdf_files



@csrf_exempt
def chat_home(request):
    # 1) Clear messages if it's a GET request
    if request.method == 'GET':
        request.session['similarity_top_k'] = 5
        request.session['chat_messages'] = []

    # 2) PDF upload logic
    if request.method == 'POST':
    # Handle multiple PDF uploads
        handle_files(request)
        chat_functionality(request)

        # Clear chat messages
        clear_chat = request.POST.get('clear_chat', '').strip()
        if clear_chat:
            logging.warning("Clearing chat messages")
            request.session['chat_messages'] = []
            
    # Gather list of PDFs in the folder
    pdf_files_list = list_pdf_files()

    return render(request, 'chat/chat_home.html', {
        'chat_messages': request.session['chat_messages'],
        'saved_int_count': request.session.get('similarity_top_k', 5),
        'pdf_files': pdf_files_list,
        'MEDIA_URL': settings.MEDIA_URL,  # So the template can reference /media/ path
    })
