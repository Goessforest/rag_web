
import os
from django.conf import settings
from ..rag._rag_defaults import RAG_defaults
import logging, threading
from pydantic import BaseModel
from django.contrib import messages
from ..rag.add_to_storage import Parse_and_Store_Vector
from ..rag.vector_retriever import VectorDBRetriever
from typing import *
# from .chat import chat_functionality

class Chat_message(BaseModel):
    role: str
    content: str

    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content
        }
    @classmethod
    def parse_from_dict(cls, message: dict):
        assert isinstance(message, dict)
        return cls(**message)
    @classmethod
    def parse_from_list(cls, messages: list) -> list:
        assert isinstance(messages, list)
        return [cls.parse_from_dict(data) for data in messages]



class Chat_home_model:
    def __init__(self, request):

        self._request = request
        self._user_id = request.user.id
        logging.warning(f"{request.method} REQUEST:USER = {request.user}; ID = {request.user.id}")

        # Create the user directory if it doesn't exist
        self._user_file_root = os.path.join(settings.MEDIA_ROOT, f'USER_{self._user_id}')
        os.makedirs(self._user_file_root, exist_ok=True)

        # Create the user's file directory if it doesn't exist
        self._users_pdf_path = os.path.join(self._user_file_root, 'pdfs')
        os.makedirs(self._users_pdf_path, exist_ok=True)

        # Create the user's markdown directory if it doesn't exist
        self._users_md_path = os.path.join(self._user_file_root, 'mds')
        os.makedirs(self._users_md_path, exist_ok=True)


        self._lock = threading.Lock()  # Global lock to protect file operations
        self._vector_store = RAG_defaults().get_vector_store(self._user_id)

        self.chat_messages = Chat_message.parse_from_list(request.session.get('chat_messages', []))

        # Actions
        if request.method == 'GET':
            self.get_request() # Handle Get Requests
        elif request.method == 'POST':
            self.post_request()


    @property
    def target_number_of_nodes(self) ->int:
        """retruns the target number of nodes, that should be retreived from the users vector store"""
        if not hasattr(self, '_target_number_of_nodes'):
            similarity_top_k = str(self._request.POST.get('similarity_top_k', 5)).strip()
            self._target_number_of_nodes = int(similarity_top_k) if similarity_top_k.isdigit() else 5
        return self._target_number_of_nodes

    def post_request(self):
        """Handle POST requests"""
        # Handle multiple PDF uploads
        self.if_file_upload()
        self.if_chat()

        # Clear chat messages
        clear_chat = self._request.POST.get('clear_chat', '').strip()
        if clear_chat:
            logging.warning("Clearing chat messages")
            self._request.session['chat_messages'] = []

    def if_chat(self):
        '''Chat functionality for the assistant'''

        user_input = self._request.POST.get('user_input', '').strip()
        if user_input:
            self.chat_messages.append(Chat_message(role="user", 
                                                   content=user_input))
            try:
                rag = VectorDBRetriever(user_id=self._user_id)
                logging.warning(f"User input: {user_input}")
                

                max_tokens = self.target_number_of_nodes * 100
                # if len(request.session['chat_messages']) == 1:
                nodes_with_scores, response = rag.query(user_input, 
                                                        similarity_top_k=self.target_number_of_nodes, 
                                                        max_tokens=max_tokens)

                references_html = ""
                for idx, node in enumerate(nodes_with_scores, start=1):
                    # Basic info
                    filename = node.metadata.get('filename', 'Not Found')
                    logging.error(node.metadata)
                    snippet_full = node.get_content().replace('\n', ' ')
                    
                    # Show only first ~80 chars in summary
                    snippet_short = snippet_full[:80]

                    references_html += f"""
                    <details>
                        <summary>({idx}) {filename} | Score: {node.score:.3f} | Snippet: "{snippet_short}..."</summary>
                        <p>{snippet_full}</p>
                    </details>
                    """
                self.chat_messages.append(Chat_message(role="source", 
                                                       content=references_html))
                # AI-generated answer
                ai_message = response.content

                # Combine references and answer into a single assistant message
                
                self.chat_messages.append(Chat_message(role="assistant", 
                                                       content= ai_message))
                self._request.session.modified = True

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.chat_messages.append(Chat_message(role="assistant", 
                                                       content= error_msg))
                self._request.session.modified = True


    def async_file_parser(self, path: str, user_id: int):
        """Background task that processes the file safely into Markdown files and adds the nodes accordingly to the vector store."""

        if not os.path.exists(path):
            logging.error(f"File {path} does not exist") 
            return
        """Background task that processes the file safely."""
        file = Parse_and_Store_Vector(path, user_id=user_id)

            # Ensure target directories exist
        with self._lock:  # Ensure only one thread modifies files at a time

            # Safe file rename operation
            extenstion = os.path.splitext(path)[1]
            os.rename(path, os.path.join(self._users_pdf_path, f"{file.name}{extenstion}"))

            # Safe markdown file writing
            with open(os.path.join(self._users_md_path, f"{file.name}.md"), "w") as f:
                f.write(file.text)


    def if_file_upload(self):
        """Handle file uploads"""

        pdf_files = self._request.FILES.getlist('pdf_files') 
        if pdf_files:
            for pdf_file in pdf_files:
                # Check if the file is a PDF

                # Step 1: Save the file
                save_path = os.path.join(self._users_pdf_path, pdf_file.name)
                with open(save_path, 'wb+') as destination:
                    for chunk in pdf_file.chunks():
                        destination.write(chunk)
                
                # Step 2: display a success message
                messages.success(self._request, f"Your PDF {pdf_file.name} was uploaded and will now be processed. This may take a few minutes!")

                
                # Step 3: Start background processing
                thread = threading.Thread(target=self.async_file_parser, args=(save_path, self._user_id), daemon=True)
                thread.start()
                logging.warning(f"Started processing {pdf_file.name} in the background.")


    def get_request(self):
        """Reset the request if it's a GET request"""
        self._target_number_of_nodes = 5
        self.chat_messages = []


    def get_pdf_files(self) -> list:
        pdf_files_list = []

        md_files_list = [os.path.basename(file).strip(".md").lower() for file in os.listdir(self._users_md_path) if file.lower().endswith('.md')]

        for f in os.listdir(self._users_pdf_path):
            if f.lower().endswith('.pdf'):
            
                file_container = {"name": f, "is_parsed": bool(os.path.basename(f).strip(".pdf").lower() in md_files_list)}
                pdf_files_list.append(file_container)
            elif f.lower().endswith('.md'):

                file_container = {"name": f, "is_parsed": bool(os.path.basename(f).strip(".md").lower() in md_files_list)}
                pdf_files_list.append(file_container)
        logging.warning(f"PDF files: {pdf_files_list}")
        return pdf_files_list


    def to_dict(self):
        """Convert the model to a dictionary"""

        logging.warning(f"{self._users_pdf_path}")
        self._request.session['chat_messages'] = [msg.to_dict() for msg in self.chat_messages]
        return {
            'chat_messages': [msg.to_dict() for msg in self.chat_messages],
            'saved_int_count': self.target_number_of_nodes,
            'pdf_files': self.get_pdf_files(),
            'MEDIA_URL': f"{settings.MEDIA_URL}/USER_{self._user_id}/",
        }