from django.http import HttpResponseBadRequest
import os
from django.conf import settings
import logging
from ..rag.add_to_storage import Parse_and_Store_Vector
import threading
import requests
from django.contrib import messages
from django.shortcuts import render



file_lock = threading.Lock()  # Global lock to protect file operations

def list_pdf_files():
    pdf_files_list = []
    pdf_folder_path = os.path.join(settings.MEDIA_ROOT, 'pdfs')
    md_folder_path = os.path.join(settings.MEDIA_ROOT, 'mds')
    md_files_list = [ os.path.basename(file).strip(".md").lower() for file in os.listdir(md_folder_path) if file.lower().endswith('.md')]
    if os.path.exists(pdf_folder_path):
        for f in os.listdir(pdf_folder_path):
            if f.lower().endswith('.pdf'):
                file_container = {"name": f, "is_parsed": bool(os.path.basename(f).strip(".pdf").lower() in md_files_list)}
                pdf_files_list.append(file_container)
    return pdf_files_list


def async_file_parser(path: str):
    """"""
    assert os.path.exists(path), "File does not exist"
    """Background task that processes the file safely."""
    file = Parse_and_Store_Vector(path)

        # Ensure target directories exist
    with file_lock:  # Ensure only one thread modifies files at a time
        pdf_folder_path = os.path.join(settings.MEDIA_ROOT, 'mds')
        os.makedirs(pdf_folder_path, exist_ok=True)

        # Safe file rename operation
        new_pdf_path = f"media/pdfs/{file.name}.pdf"
        os.rename(file.path, new_pdf_path)

        # Safe markdown file writing
        md_path = f"media/mds/{file.name}.md"
        with open(md_path, "w") as f:
            f.write(file.text)

    # Notify the callback URL
    try:
        # messages.success(request, f"Your PDF '{file.name}' is now available for search.")
        logging.warning(f"File '{file.name}' processed successfully.")
    except requests.RequestException as e:
        # messages.success(request, f"Your PDF '{file.name}' is now available for search.")
        logging.error(f"File '{file.name}' failed to process.")
        print(f"Error sending callback: {e}")
    # finally:
    #     render(request, 'chat/chat_home.html', )



def handle_files(request):
    pdf_files = request.FILES.getlist('pdf_files') 
    if pdf_files:
        # create a folder for PDFS
        pdf_folder_path = os.path.join(settings.MEDIA_ROOT, 'pdfs')
        os.makedirs(pdf_folder_path, exist_ok=True)


        for pdf_file in pdf_files:
            if not pdf_file.name.lower().endswith('.pdf'):
                return HttpResponseBadRequest("All files must be PDFs.")

            # upload file
            save_path = os.path.join(pdf_folder_path, pdf_file.name)
            with open(save_path, 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)
            
            # display Uploeaded message
            messages.success(request, f"Your PDF {pdf_file.name} was uploaded and will now be processed. This may take a few minutes!")

            
                    # Start background processing
            thread = threading.Thread(target=async_file_parser, args=(save_path), daemon=True)
            thread.start()
            logging.warning(f"Started processing {pdf_file.name} in the background.")
            