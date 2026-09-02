"""
Semantic analysis agent for evaluation.

This agent demonstrates semantic analysis capabilities including
sentiment analysis, text classification, and semantic similarity.
"""

import time
from typing import Dict, Any


class SemanticAnalysisAgent:
    """Semantic analysis agent for various NLP tasks."""
    
    def __init__(self, name: str = "semantic_analysis_agent"):
        self.name = name
        self._is_initialized = False
        
        # Define sentiment keywords
        self.positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy', 'perfect']
        self.negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad', 'disappointed', 'worst', 'poor']
        
        # Define category keywords
        self.category_keywords = {
            'technology': ['computer', 'software', 'app', 'digital', 'online', 'internet', 'code', 'programming'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'match', 'championship', 'athlete'],
            'food': ['restaurant', 'meal', 'recipe', 'cook', 'delicious', 'taste', 'flavor', 'ingredient'],
            'travel': ['trip', 'vacation', 'hotel', 'flight', 'destination', 'journey', 'explore', 'adventure']
        }
    
    def initialize(self, **kwargs) -> None:
        """Initialize the agent."""
        self._is_initialized = True
        print(f"Semantic analysis agent '{self.name}' initialized successfully")
    
    def predict(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Make semantic analysis prediction.
        
        Args:
            input_text: Input text to analyze
            **kwargs: Additional parameters including task_type, text2, categories
            
        Returns:
            Dictionary containing prediction results
        """
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        start_time = time.time()
        task_type = kwargs.get('task_type', 'sentiment')
        
        if task_type == 'sentiment':
            result = self._analyze_sentiment(input_text)
        elif task_type == 'classification':
            categories = kwargs.get('categories', ['technology', 'sports', 'food', 'travel'])
            result = self._classify_text(input_text, categories)
        elif task_type == 'similarity':
            text2 = kwargs.get('text2', '')
            result = self._semantic_similarity(input_text, text2)
        else:
            result = {"error": f"Unknown task type: {task_type}"}
        
        processing_time = time.time() - start_time
        
        return {
            'output': result.get('analysis', str(result)),
            'confidence': result.get('confidence', 0.5),
            'processing_time': processing_time,
            'metadata': {
                'task_type': task_type,
                'agent_name': self.name,
                'method': 'rule_based_analysis',
                **{k: v for k, v in result.items() if k not in ['analysis', 'confidence']}
            }
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text with improved handling of negation and nuance."""
        text_lower = text.lower()
        
        # Enhanced sentiment word lists with intensity modifiers
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy', 'joy', 'perfect', 'awesome', 'brilliant']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'sad', 'disappointed', 'worst', 'poor', 'disgusting', 'pathetic']
        
        # Negation words that flip sentiment
        negation_words = ['not', 'no', 'never', "n't", 'hardly', 'barely', 'scarcely']
        
        # Intensity modifiers
        intensifiers = ['very', 'extremely', 'really', 'absolutely', 'completely', 'totally']
        diminishers = ['slightly', 'somewhat', 'rather', 'quite', 'kinda', 'sort of']
        
        # Count words
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        negation_count = sum(1 for word in negation_words if word in text_lower)
        intensifier_count = sum(1 for word in intensifiers if word in text_lower)
        diminisher_count = sum(1 for word in diminishers if word in text_lower)
        
        # Handle negation - flip sentiment if negation is present
        if negation_count > 0:
            if positive_count > negative_count:
                # "not good" becomes negative
                negative_count += positive_count * 0.8
                positive_count *= 0.2
            elif negative_count > positive_count:
                # "not bad" becomes positive  
                positive_count += negative_count * 0.8
                negative_count *= 0.2
        
        # Apply intensity modifiers
        intensity_multiplier = 1.0 + (intensifier_count * 0.3) - (diminisher_count * 0.2)
        positive_count *= intensity_multiplier
        negative_count *= intensity_multiplier
        
        # Determine sentiment with better thresholds
        diff = positive_count - negative_count
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = 'neutral'
            confidence = 0.4  # Low confidence when no sentiment words found
        elif abs(diff) <= 0.5:  # Very close counts
            sentiment = 'neutral'
            confidence = 0.6
        elif diff > 0:
            sentiment = 'positive'
            confidence = min(0.95, 0.6 + (diff / total_sentiment_words) * 0.4)
        else:
            sentiment = 'negative'
            confidence = min(0.95, 0.6 + (abs(diff) / total_sentiment_words) * 0.4)
            
        # Adjust confidence based on text complexity
        word_count = len(text.split())
        if word_count > 15:
            confidence *= 0.9  # Slightly reduce confidence for longer, more complex sentences
            
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'analysis': f"Sentiment: {sentiment} (confidence: {confidence:.2f}) - {positive_count:.1f} positive, {negative_count:.1f} negative, {negation_count} negations"
        }
    
    def _classify_text(self, text: str, categories: list) -> Dict[str, Any]:
        """Classify text into given categories with improved scoring."""
        text_lower = text.lower()
        scores = {}
        
        # Enhanced category keywords with more specific terms
        enhanced_keywords = {
            'technology': ['computer', 'software', 'app', 'digital', 'online', 'internet', 'code', 'programming', 
                          'algorithm', 'data', 'system', 'network', 'website', 'interface', 'update', 'performance'],
            'sports': ['game', 'team', 'player', 'score', 'win', 'match', 'championship', 'athlete',
                       'football', 'basketball', 'soccer', 'tennis', 'coach', 'season', 'league', 'tournament'],
            'food': ['restaurant', 'meal', 'recipe', 'cook', 'delicious', 'taste', 'flavor', 'ingredient',
                     'kitchen', 'dining', 'cuisine', 'dish', 'fresh', 'baking', 'grill', 'menu'],
            'travel': ['trip', 'vacation', 'hotel', 'flight', 'destination', 'journey', 'explore', 'adventure',
                       'airport', 'passport', 'booking', 'resort', 'tourism', 'museum', 'sightseeing', 'abroad']
        }
        
        for category in categories:
            keywords = enhanced_keywords.get(category, self.category_keywords.get(category, []))
            if not keywords:
                scores[category] = 0
                continue
                
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            # Calculate score with normalization and boost for exact matches
            base_score = matches / len(keywords)
            
            # Boost score for multiple matches
            if matches > 1:
                base_score *= (1 + (matches - 1) * 0.2)
            
            # Apply length penalty for very short texts
            word_count = len(text.split())
            if word_count < 5:
                base_score *= 0.7
                
            scores[category] = min(1.0, base_score)
        
        # Find best category
        if not scores or max(scores.values()) == 0:
            return {
                'category': 'unknown',
                'confidence': 0.2,
                'all_scores': scores,
                'analysis': 'Category: unknown (no matching keywords found)'
            }
        
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        # Reduce confidence if scores are too close (ambiguous case)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 0.1:
            confidence *= 0.7
            
        return {
            'category': best_category,
            'confidence': confidence,
            'all_scores': scores,
            'analysis': f"Category: {best_category} (confidence: {confidence:.2f})"
        }
    
    def _semantic_similarity(self, text1: str, text2: str) -> Dict[str, Any]:
        """Calculate semantic similarity between two texts with improved metrics."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return {'similarity': 0.0, 'common_words': 0, 'analysis': 'No similarity - empty text'}
        
        # Basic Jaccard similarity
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        jaccard_similarity = len(common_words) / len(total_words)
        
        # Additional similarity metrics
        # Word overlap ratio (more lenient than Jaccard)
        overlap_ratio = len(common_words) / min(len(words1), len(words2))
        
        # Length similarity (penalize very different lengths)
        len1, len2 = len(words1), len(words2)
        length_similarity = 1 - abs(len1 - len2) / max(len1, len2)
        
        # Combined similarity score
        combined_similarity = (jaccard_similarity * 0.6) + (overlap_ratio * 0.3) + (length_similarity * 0.1)
        
        # Categorize similarity level
        if combined_similarity > 0.7:
            similarity_level = "high"
        elif combined_similarity > 0.3:
            similarity_level = "moderate"
        else:
            similarity_level = "low"
        
        return {
            'similarity': combined_similarity,
            'jaccard_similarity': jaccard_similarity,
            'overlap_ratio': overlap_ratio,
            'common_words': len(common_words),
            'common_words_list': list(common_words),
            'analysis': f"Similarity: {combined_similarity:.3f} ({similarity_level}) - {len(common_words)} common words"
        }


class AlternativeSemanticAgent:
    """Alternative semantic analysis agent with different approach."""
    
    def __init__(self, name: str = "alternative_semantic_agent"):
        self.name = name
        self._is_initialized = False
    
    def initialize(self, **kwargs) -> None:
        """Initialize the alternative agent."""
        self._is_initialized = True
        print(f"Alternative semantic agent '{self.name}' initialized")
    
    def predict(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Alternative prediction approach."""
        if not self._is_initialized:
            raise RuntimeError("Agent not initialized")
        
        # Simple word count based analysis
        words = input_text.lower().split()
        
        task_type = kwargs.get('task_type', 'sentiment')
        
        if task_type == 'sentiment':
            # Very simple sentiment based on exclamation marks and length
            if '!' in input_text and len(words) > 5:
                sentiment = 'positive'
                confidence = 0.7
            elif len(words) < 5:
                sentiment = 'neutral'
                confidence = 0.5
            else:
                sentiment = 'negative'
                confidence = 0.6
                
            analysis = f"Sentiment: {sentiment} (simple analysis)"
            
        else:
            # Default response for other tasks
            analysis = f"Processed {task_type} task with simple method"
            confidence = 0.5
        
        return {
            'output': analysis,
            'confidence': confidence,
            'processing_time': 0.02,
            'metadata': {
                'method': 'simple_analysis',
                'word_count': len(words)
            }
        }
