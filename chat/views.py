# chat/views.py
import os
# import openai
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.http import HttpResponseBadRequest
from .rag import setup_rag
import logging

# Set up the RAG model
rag = setup_rag(number_of_results=5)

# Set your API key from environment or Django settings
# openai.api_key = os.environ.get('OPENAI_API_KEY') or getattr(settings, 'OPENAI_API_KEY', None)

@csrf_exempt
def chat_home(request):
    # 1) Clear messages if it's a GET request
    if request.method == 'GET':
        request.session['messages'] = []

    # PDF folder path
    pdf_folder_path = os.path.join(settings.MEDIA_ROOT, 'pdfs')
    os.makedirs(pdf_folder_path, exist_ok=True)


    # 2) PDF upload logic
    if request.method == 'POST':
        # 1) Handle multiple PDF uploads
        pdf_files = request.FILES.getlist('pdf_files')  # matches <input name="pdf_files" multiple>
        for pdf_file in pdf_files:
            if not pdf_file.name.lower().endswith('.pdf'):
                return HttpResponseBadRequest("All files must be PDFs.")

            save_path = os.path.join(pdf_folder_path, pdf_file.name)
            with open(save_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)


        # 3) Chat message logic
    user_input = request.POST.get('user_input', '').strip()
    if user_input:
        # Log the user's message and store it
        request.session['messages'].append({"role": "user", "content": user_input})

        try:
            logging.warning(f"User input: {user_input}")
            
            # Query your RAG or LLM system
            nodes_with_scores, response = rag.query(user_input)
            
            # Build an HTML string containing references (filename + score + short snippet)
            references_html = ""
            for idx, node in enumerate(nodes_with_scores, start=1):
                # Basic info
                filename = node.metadata.get('filename', 'Not Found')
                snippet_full = node.get_content().replace('\n', ' ')
                
                # Show only first ~80 chars in summary
                snippet_short = snippet_full[:80]

                references_html += f"""
                <details>
                    <summary>({idx}) {filename} | Score: {node.score:.3f} | Snippet: "{snippet_short}..."</summary>
                    <p>{snippet_full}</p>
                </details>
                """
            request.session['messages'].append({"role": "source", "content": references_html})
            # AI-generated answer
            ai_message = response.choices[0].message.content

            # Combine references and answer into a single assistant message
            
            request.session['messages'].append({"role": "assistant", "content": ai_message})
            request.session.modified = True

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            request.session['messages'].append({"role": "assistant", "content": error_msg})
            request.session.modified = True

    # Gather list of PDFs in the folder
    pdf_files_list = []
    if os.path.exists(pdf_folder_path):
        for f in os.listdir(pdf_folder_path):
            if f.lower().endswith('.pdf'):
                pdf_files_list.append(f)

    return render(request, 'chat/chat_home.html', {
        'messages': request.session['messages'],
        'pdf_files': pdf_files_list,
        'MEDIA_URL': settings.MEDIA_URL,  # So the template can reference /media/ path
    })
