import logging
from django.http import HttpResponse

logger = logging.getLogger(__name__)

def test_view(request):
    origin = request.META.get('HTTP_ORIGIN', 'No Origin header')
    # Retrieve CSRF token from the cookies and POST data (if present)
    csrf_token_cookie = request.COOKIES.get('csrftoken', 'No CSRF token in cookie')
    csrf_token_post = request.POST.get('csrfmiddlewaretoken', 'No CSRF token in POST data')
    
    logger.warning(
        f"Request Origin: {origin}, CSRF token (cookie): {csrf_token_cookie}, CSRF token (POST): {csrf_token_post}"
    )
    return HttpResponse("OK")
