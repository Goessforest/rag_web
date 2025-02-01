import logging
from ..rag.vector_retriever import VectorDBRetriever


def chat_functionality(request):
    '''Chat functionality for the assistant'''

    user_input = request.POST.get('user_input', '').strip()
    if user_input:
        request.session['chat_messages'].append({"role": "user", "content": user_input})
        try:
            rag = VectorDBRetriever()
            logging.warning(f"User input: {user_input}")
            
            # Query your RAG or LLM system
            similarity_top_k = str(request.POST.get('similarity_top_k', 5).strip())
            similarity_top_k = int(similarity_top_k) if similarity_top_k.isdigit() else 5
            request.session['similarity_top_k'] = int(similarity_top_k)

            max_tokens = similarity_top_k * 100
            # if len(request.session['chat_messages']) == 1:
            nodes_with_scores, response = rag.query(user_input, 
                                                    similarity_top_k=similarity_top_k, 
                                                    max_tokens=max_tokens)
            # else:
            #     max_tokens += len(request.session['chat_messages']) * 1000
            #     nodes_with_scores, response = rag.follow_up_query(user_input, 
            #                                             similarity_top_k=similarity_top_k, 
            #                                             max_tokens=max_tokens)
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
            request.session['chat_messages'].append({"role": "source", "content": references_html})
            # AI-generated answer
            ai_message = response.content

            # Combine references and answer into a single assistant message
            
            request.session['chat_messages'].append({"role": "assistant", "content": ai_message})
            request.session.modified = True

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            request.session['chat_messages'].append({"role": "assistant", "content": error_msg})
            request.session.modified = True