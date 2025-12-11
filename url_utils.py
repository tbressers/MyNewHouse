"""
URL normalization utilities to avoid duplicate links.
"""

from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


def normalize_url(url: str) -> str:
    """
    Normalize a URL to ensure consistent comparison.
    
    - Removes trailing slashes
    - Converts to lowercase domain
    - Sorts query parameters
    - Removes fragments
    - Removes default ports (80, 443)
    
    Args:
        url: The URL to normalize
        
    Returns:
        Normalized URL string
    """
    if not url:
        return url
    
    parsed = urlparse(url)
    
    # Normalize scheme to lowercase
    scheme = parsed.scheme.lower()
    
    # Normalize netloc (domain) to lowercase, remove default ports
    netloc = parsed.netloc.lower()
    if netloc.endswith(':80') and scheme == 'http':
        netloc = netloc[:-3]
    elif netloc.endswith(':443') and scheme == 'https':
        netloc = netloc[:-4]
    
    # Normalize path - remove trailing slash (except for root)
    path = parsed.path
    if path and path != '/' and path.endswith('/'):
        path = path.rstrip('/')
    
    # Sort query parameters for consistent ordering
    query = parsed.query
    if query:
        params = parse_qs(query, keep_blank_values=True)
        # Sort and rebuild query string
        sorted_params = sorted(params.items())
        query = urlencode(sorted_params, doseq=True)
    
    # Remove fragment (not relevant for identifying unique pages)
    fragment = ''
    
    # Rebuild URL
    normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
    
    return normalized
