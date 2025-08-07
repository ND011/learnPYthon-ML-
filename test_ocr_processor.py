"""
Unit tests for ResumeOCRProcessor class.

Tests enhanced OCR functionality, image preprocessing, confidence scoring,
and resume-specific text processing.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import numpy as np

# Import the classes to test
try:
    from resume_keyword_extractor.processors.ocr_processor import (
        ResumeOCRProcessor, 
        ImageQuality, 
        OCRBlock, 
        OCRResult
    )
except ImportError:
    # Skip tests if dependencies not available
    ResumeOCRProcessor = None


@unittest.skipIf(ResumeOCRProcessor is None, "OCR processor dependencies not available")
class TestResumeOCRProcessor(unittest.TestCase):
    """Test cases for ResumeOCRProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ResumeOCRProcessor()
    
    def test_init(self):
        """Test ResumeOCRProcessor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.languages, ['en'])
        self.assertIn('email', self.processor.resume_patterns)
        self.assertIn('phone', self.processor.resume_patterns)
        self.assertIn('high', self.processor.confidence_thresholds)
    
    def test_init_with_languages(self):
        """Test initialization with custom languages."""
        processor = ResumeOCRProcessor(['en', 'es'])
        self.assertEqual(processor.languages, ['en', 'es'])
    
    def test_classify_text_block_section_header(self):
        """Test classification of section headers."""
        bbox = [(0, 0), (100, 0), (100, 20), (0, 20)]
        
        # Test various section headers
        headers = ['EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS', 'SUMMARY']
        for header in headers:
            block_type = self.processor._classify_text_block(header, bbox)
            self.assertEqual(block_type, 'section_header')
    
    def test_classify_text_block_contact_info(self):
        """Test classification of contact information."""
        bbox = [(0, 0), (200, 0), (200, 20), (0, 20)]
        
        # Test email
        block_type = self.processor._classify_text_block('john.doe@email.com', bbox)
        self.assertEqual(block_type, 'contact_info')
        
        # Test phone
        block_type = self.processor._classify_text_block('(555) 123-4567', bbox)
        self.assertEqual(block_type, 'contact_info')
        
        # Test LinkedIn
        block_type = self.processor._classify_text_block('linkedin.com/in/johndoe', bbox)
        self.assertEqual(block_type, 'contact_info')
        
        # Test GitHub
        block_type = self.processor._classify_text_block('github.com/johndoe', bbox)
        self.assertEqual(block_type, 'contact_info')
    
    def test_classify_text_block_dates(self):
        """Test classification of date information."""
        bbox = [(0, 0), (100, 0), (100, 20), (0, 20)]
        
        dates = ['Jan 2020', 'February 2021', '01/2020', '2019 - 2021']
        for date in dates:
            block_type = self.processor._classify_text_block(date, bbox)
            self.assertEqual(block_type, 'date_info')
    
    def test_classify_text_block_bullet_points(self):
        """Test classification of bullet points."""
        bbox = [(0, 0), (200, 0), (200, 20), (0, 20)]
        
        bullets = ['• Developed software', '▪ Led team of 5', '‣ Improved performance']
        for bullet in bullets:
            block_type = self.processor._classify_text_block(bullet, bbox)
            self.assertEqual(block_type, 'bullet_point')
    
    def test_classify_text_block_name(self):
        """Test classification of names."""
        # Name should be at top of image (low y-coordinate)
        bbox = [(0, 10), (150, 10), (150, 30), (0, 30)]
        
        block_type = self.processor._classify_text_block('John Doe', bbox)
        self.assertEqual(block_type, 'name')
        
        # Same text lower in image should not be classified as name
        bbox_lower = [(0, 200), (150, 200), (150, 220), (0, 220)]
        block_type = self.processor._classify_text_block('John Doe', bbox_lower)
        self.assertEqual(block_type, 'text')
    
    def test_classify_text_block_default(self):
        """Test default text classification."""
        bbox = [(0, 100), (200, 100), (200, 120), (0, 120)]
        
        block_type = self.processor._classify_text_block('Regular text content', bbox)
        self.assertEqual(block_type, 'text')
    
    @patch('numpy.array')
    def test_assess_image_quality_excellent(self, mock_array):
        """Test image quality assessment for excellent quality."""
        # Mock high resolution, good contrast image
        mock_img = np.zeros((1000, 800), dtype=np.uint8)
        mock_img[:400, :] = 50   # Dark area
        mock_img[400:, :] = 200  # Light area
        mock_array.return_value = mock_img
        
        mock_image = Mock()
        mock_image.convert.return_value = mock_image
        
        quality = self.processor._assess_image_quality(mock_image)
        self.assertEqual(quality, ImageQuality.EXCELLENT)
    
    @patch('numpy.array')
    def test_assess_image_quality_poor(self, mock_array):
        """Test image quality assessment for poor quality."""
        # Mock low resolution, low contrast image
        mock_img = np.full((200, 150), 128, dtype=np.uint8)  # Low contrast, small size
        mock_array.return_value = mock_img
        
        mock_image = Mock()
        mock_image.convert.return_value = mock_image
        
        quality = self.processor._assess_image_quality(mock_image)
        self.assertEqual(quality, ImageQuality.POOR)
    
    def test_clean_text_line(self):
        """Test text line cleaning functionality."""
        # Test whitespace cleanup
        cleaned = self.processor._clean_text_line('  Multiple   spaces   here  ')
        self.assertEqual(cleaned, 'Multiple spaces here')
        
        # Test punctuation fixes
        cleaned = self.processor._clean_text_line('Text ,with bad spacing .')
        self.assertEqual(cleaned, 'Text,with bad spacing.')
        
        # Test parentheses fixes
        cleaned = self.processor._clean_text_line('Text ( with ) parentheses')
        self.assertEqual(cleaned, 'Text (with) parentheses')
    
    def test_final_text_cleanup(self):
        """Test final text cleanup functionality."""
        # Test excessive blank line removal
        text = "Line 1\n\n\n\nLine 2\n\n\n\nLine 3"
        cleaned = self.processor._final_text_cleanup(text)
        self.assertNotIn('\n\n\n', cleaned)
        
        # Test bullet point cleanup
        text = "Regular text\n   • Bullet point\n    ▪ Another bullet"
        cleaned = self.processor._final_text_cleanup(text)
        self.assertIn('\n• Bullet point', cleaned)
        self.assertIn('\n• Another bullet', cleaned)
    
    def test_calculate_overall_confidence_empty(self):
        """Test confidence calculation with empty blocks."""
        confidence = self.processor._calculate_overall_confidence([])
        self.assertEqual(confidence, 0.0)
    
    def test_calculate_overall_confidence_weighted(self):
        """Test weighted confidence calculation."""
        blocks = [
            OCRBlock("Short", 0.9, []),
            OCRBlock("Much longer text content", 0.7, [])
        ]
        
        confidence = self.processor._calculate_overall_confidence(blocks)
        
        # Should be weighted toward the longer text
        self.assertLess(confidence, 0.9)  # Less than high confidence short text
        self.assertGreater(confidence, 0.7)  # Greater than low confidence long text
    
    def test_get_confidence_assessment(self):
        """Test confidence assessment descriptions."""
        # High confidence
        result = OCRResult("", [], 0.85, ImageQuality.GOOD, [], {})
        assessment = self.processor.get_confidence_assessment(result)
        self.assertIn("High", assessment)
        
        # Medium confidence
        result = OCRResult("", [], 0.65, ImageQuality.GOOD, [], {})
        assessment = self.processor.get_confidence_assessment(result)
        self.assertIn("Medium", assessment)
        
        # Low confidence
        result = OCRResult("", [], 0.35, ImageQuality.GOOD, [], {})
        assessment = self.processor.get_confidence_assessment(result)
        self.assertIn("Low", assessment)
    
    def test_get_improvement_suggestions_poor_quality(self):
        """Test improvement suggestions for poor quality images."""
        result = OCRResult("", [], 0.5, ImageQuality.POOR, [], {})
        suggestions = self.processor.get_improvement_suggestions(result)
        
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("resolution" in s.lower() for s in suggestions))
        self.assertTrue(any("lighting" in s.lower() for s in suggestions))
    
    def test_get_improvement_suggestions_low_confidence(self):
        """Test improvement suggestions for low confidence results."""
        result = OCRResult("", [], 0.3, ImageQuality.FAIR, [], {})
        suggestions = self.processor.get_improvement_suggestions(result)
        
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("rotated" in s.lower() for s in suggestions))
    
    def test_get_improvement_suggestions_few_blocks(self):
        """Test improvement suggestions for results with few text blocks."""
        result = OCRResult("", [OCRBlock("test", 0.8, [])], 0.8, ImageQuality.GOOD, [], {})
        suggestions = self.processor.get_improvement_suggestions(result)
        
        self.assertTrue(any("complete resume" in s.lower() for s in suggestions))
    
    def test_ocr_block_dataclass(self):
        """Test OCRBlock dataclass."""
        bbox = [(0, 0), (100, 0), (100, 20), (0, 20)]
        block = OCRBlock("Test text", 0.95, bbox, "header")
        
        self.assertEqual(block.text, "Test text")
        self.assertEqual(block.confidence, 0.95)
        self.assertEqual(block.bbox, bbox)
        self.assertEqual(block.block_type, "header")
    
    def test_ocr_result_dataclass(self):
        """Test OCRResult dataclass."""
        blocks = [OCRBlock("Test", 0.9, [])]
        metadata = {"test": "value"}
        
        result = OCRResult(
            text="Test text",
            blocks=blocks,
            overall_confidence=0.85,
            image_quality=ImageQuality.GOOD,
            preprocessing_applied=["contrast"],
            metadata=metadata
        )
        
        self.assertEqual(result.text, "Test text")
        self.assertEqual(result.blocks, blocks)
        self.assertEqual(result.overall_confidence, 0.85)
        self.assertEqual(result.image_quality, ImageQuality.GOOD)
        self.assertEqual(result.preprocessing_applied, ["contrast"])
        self.assertEqual(result.metadata, metadata)


class TestOCRProcessorBasic(unittest.TestCase):
    """Basic tests that don't require external dependencies."""
    
    def test_image_quality_enum(self):
        """Test ImageQuality enum values."""
        self.assertEqual(ImageQuality.EXCELLENT.value, "excellent")
        self.assertEqual(ImageQuality.GOOD.value, "good")
        self.assertEqual(ImageQuality.FAIR.value, "fair")
        self.assertEqual(ImageQuality.POOR.value, "poor")


if __name__ == '__main__':
    unittest.main()