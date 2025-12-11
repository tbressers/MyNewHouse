"""
Intelligent classifier for property listings without predefined patterns.
Uses URL structure analysis and content heuristics to identify actual property pages.
"""

import re
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class URLAnalyzer:
    """Analyzes URL structure to distinguish property listings from overview/category pages."""
    
    # Generic patterns that indicate it's NOT a property listing
    ANTI_PATTERNS = [
        # Pagination indicators
        (r'[?&]page=\d+', 'pagination_param'),
        (r'[?&]start=\d+', 'pagination_param'),
        (r'/page/\d+', 'pagination_path'),
        
        # Category/filter endings (too generic)
        (r'/(?:kamers?|rooms?|appartement(?:en)?|apartments?|studio(?:s)?|woningen?|houses?)/?$', 'category_ending'),
        
        # Search/overview pages
        (r'-(?:overzicht|overview|search|zoeken)', 'overview_keyword'),
        (r'/(?:for-rent|te-huur|huurwoningen)/?$', 'generic_rental_root'),
        
        # Geographic filters without specific property
        (r'/wijk-[^/]+/?$', 'district_only'),
        (r'/(?:city|stad)-[^/]+/?$', 'city_only'),
    ]
    
    # Patterns that strongly suggest it IS a property listing
    POSITIVE_PATTERNS = [
        # Specific address patterns in URLs
        (r'/[a-z\-]+\d+[a-z]?(?:-\d+)?(?:/|$)', 'street_with_number', 3),
        
        # Property IDs (numbers/codes that look like unique identifiers)
        (r'/(?:property|listing|object|woning|kamer)-(?:\d{5,}|[a-f0-9]{8,})', 'unique_id_prefix', 4),
        (r'/\d{5,}(?:/|$)', 'numeric_id', 2),
        (r'/[a-z]{2}\d{6,}', 'code_id', 3),
        
        # "te-huur" (for rent) with specific location
        (r'/[a-z\-]+-te-huur/[a-z\-]+\d+', 'te_huur_with_address', 4),
        
        # Date patterns (often used for room listings)
        (r'/\d{4}-\d{2}-\d{2}(?:/|$)', 'date_pattern', 3),
        
        # Multiple path segments (likely specific property, not category)
        (r'^[^?#]*(/[^/?#]+){4,}', 'deep_path', 2),
    ]
    
    @classmethod
    def analyze_url(cls, url: str) -> Tuple[int, Dict[str, any]]:
        """
        Analyze URL and return a score indicating likelihood it's a property listing.
        
        Returns:
            Tuple of (score, details_dict)
            - score > 0: likely property listing
            - score < 0: likely overview/category page
            - score = 0: uncertain
        """
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        full_url = url.lower()
        
        score = 0
        details = {'positive': [], 'negative': []}
        
        # Check anti-patterns (negative indicators)
        for pattern, label in cls.ANTI_PATTERNS:
            if re.search(pattern, full_url, re.IGNORECASE):
                score -= 2
                details['negative'].append(label)
        
        # Check positive patterns (with weights)
        for pattern, label, weight in cls.POSITIVE_PATTERNS:
            if re.search(pattern, full_url, re.IGNORECASE):
                score += weight
                details['positive'].append(label)
        
        # Path depth analysis: deeper paths are more likely to be specific properties
        path_segments = [s for s in path.split('/') if s and s not in ('en', 'nl', 'nl-nl', 'en-us')]
        if len(path_segments) >= 4:
            score += 1
            details['positive'].append('deep_path')
        elif len(path_segments) <= 1:
            score -= 1
            details['negative'].append('shallow_path')
        
        # Query string complexity: simple filters suggest overview pages
        query_params = parse_qs(query)
        if any(key in query for key in ['city', 'type', 'category', 'filter']):
            score -= 1
            details['negative'].append('filter_params')
        
        # Slug analysis: URLs with meaningful slugs (not just IDs) are often listings
        last_segment = path_segments[-1] if path_segments else ''
        if len(last_segment) > 15 and '-' in last_segment:
            # Looks like a descriptive slug
            score += 1
            details['positive'].append('descriptive_slug')
        
        return score, details


class ContentAnalyzer:
    """Analyzes link text and context to identify property listings."""
    
    # Strong indicators of property content
    PROPERTY_INDICATORS = [
        (r'\b\d+\s*m²', 'size_sqm', 3),
        (r'€\s*[\d,\.]+', 'price_euro', 3),
        (r'\b\d+\s*(?:kamers?|rooms?|bedrooms?)\b', 'room_count', 2),
        (r'\b(?:beschikbaar|available)\b.*?\d{1,2}[-/]\d{1,2}', 'availability_date', 2),
        (r'\b(?:gemeubileerd|furnished|unfurnished|ongemeubileerd)\b', 'furnishing', 2),
        (r'\b(?:balkon|balcony|terras|terrace|tuin|garden)\b', 'amenity', 1),
    ]
    
    # Address patterns (street name + number)
    ADDRESS_PATTERNS = [
        (r'\b[A-Z][a-z]+(?:straat|laan|weg|plein|singel|gracht|kade)\s+\d+', 'dutch_street', 4),
        (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d+', 'named_street', 3),
        (r'\b\d{4}\s*[A-Z]{2}\b', 'postal_code', 2),
    ]
    
    # City names
    CITIES = ['Nijmegen', 'Amsterdam', 'Utrecht', 'Rotterdam', 'Den Haag', 'Eindhoven', 'Groningen']
    
    @classmethod
    def analyze_content(cls, title: str, url: str = '') -> Tuple[int, Dict[str, any]]:
        """
        Analyze link text/title and return a score indicating property listing likelihood.
        
        Returns:
            Tuple of (score, details_dict)
        """
        if not title:
            return 0, {'reason': 'no_title'}
        
        score = 0
        details = {'indicators': []}
        
        # Check property indicators
        for pattern, label, weight in cls.PROPERTY_INDICATORS:
            if re.search(pattern, title, re.IGNORECASE):
                score += weight
                details['indicators'].append(label)
        
        # Check address patterns
        for pattern, label, weight in cls.ADDRESS_PATTERNS:
            if re.search(pattern, title):
                score += weight
                details['indicators'].append(label)
        
        # Check for city names
        for city in cls.CITIES:
            if city.lower() in title.lower():
                score += 1
                details['indicators'].append('city_name')
                break
        
        # Title length heuristic: very short titles are often navigation links
        if len(title) < 10:
            score -= 1
            details['indicators'].append('too_short')
        elif len(title) > 30:
            # Longer, descriptive titles are often property listings
            score += 1
            details['indicators'].append('descriptive_length')
        
        # Check for navigation-type words (negative indicators)
        nav_words = r'\b(?:home|menu|contact|about|over ons|login|register|zoeken|search|filter|sort|toon|show all)\b'
        if re.search(nav_words, title, re.IGNORECASE):
            score -= 2
            details['indicators'].append('navigation_word')
        
        return score, details


class IntelligentClassifier:
    """
    Combined classifier that uses both URL structure and content analysis
    to identify property listings without predefined exclusion lists.
    """
    
    def __init__(self, url_weight: float = 0.6, content_weight: float = 0.4):
        """
        Initialize classifier with configurable weights.
        
        Args:
            url_weight: Weight for URL analysis (0-1)
            content_weight: Weight for content analysis (0-1)
        """
        self.url_weight = url_weight
        self.content_weight = content_weight
        self.url_analyzer = URLAnalyzer()
        self.content_analyzer = ContentAnalyzer()
        self.stats = Counter()
    
    def classify_link(self, url: str, title: str = '', threshold: float = 2.0) -> Tuple[bool, float, Dict]:
        """
        Classify whether a link is a property listing.
        
        Args:
            url: The URL to classify
            title: Optional link text/title
            threshold: Minimum score to consider it a property listing
        
        Returns:
            Tuple of (is_property_listing, confidence_score, details)
        """
        # Analyze URL
        url_score, url_details = self.url_analyzer.analyze_url(url)
        
        # Analyze content
        content_score, content_details = self.content_analyzer.analyze_content(title, url)
        
        # Combine scores with weights
        combined_score = (url_score * self.url_weight) + (content_score * self.content_weight)
        
        is_listing = combined_score >= threshold
        
        details = {
            'url_score': url_score,
            'content_score': content_score,
            'combined_score': combined_score,
            'url_details': url_details,
            'content_details': content_details,
            'threshold': threshold
        }
        
        # Update stats
        if is_listing:
            self.stats['classified_as_listing'] += 1
        else:
            self.stats['classified_as_non_listing'] += 1
        
        return is_listing, combined_score, details
    
    def batch_classify(self, links: List[Dict], threshold: float = 2.0) -> List[Dict]:
        """
        Classify a batch of links and return only property listings.
        
        Args:
            links: List of link dictionaries with 'link' and optionally 'title'
            threshold: Minimum score to consider it a property listing
        
        Returns:
            Filtered list of property listings with classification details
        """
        results = []
        
        for link_data in links:
            url = link_data.get('link', '')
            title = link_data.get('title', '')
            
            if not url:
                continue
            
            is_listing, score, details = self.classify_link(url, title, threshold)
            
            if is_listing:
                # Add classification metadata
                link_data['classification'] = {
                    'score': score,
                    'details': details
                }
                results.append(link_data)
        
        logger.info(f"Classified {len(links)} links -> {len(results)} property listings (threshold={threshold})")
        logger.info(f"Classification stats: {dict(self.stats)}")
        
        return results
    
    def auto_tune_threshold(self, links: List[Dict], expected_min: int = 1, expected_max: int = 100) -> float:
        """
        Automatically tune the threshold based on the distribution of scores.
        
        Args:
            links: Sample links to analyze
            expected_min: Minimum expected property listings
            expected_max: Maximum expected property listings
        
        Returns:
            Optimal threshold value
        """
        scores = []
        for link_data in links:
            url = link_data.get('link', '')
            title = link_data.get('title', '')
            _, score, _ = self.classify_link(url, title, threshold=0)  # Get raw score
            scores.append(score)
        
        if not scores:
            return 2.0  # Default
        
        scores.sort(reverse=True)
        
        # Find threshold where we get between expected_min and expected_max listings
        for threshold in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            count = sum(1 for s in scores if s >= threshold)
            if expected_min <= count <= expected_max:
                logger.info(f"Auto-tuned threshold to {threshold} (yields {count} listings)")
                return threshold
        
        # Fallback: use median of positive scores
        positive_scores = [s for s in scores if s > 0]
        if positive_scores:
            threshold = sorted(positive_scores)[len(positive_scores) // 2]
            logger.info(f"Auto-tuned threshold to {threshold} (median of positive scores)")
            return threshold
        
        return 2.0  # Default fallback
