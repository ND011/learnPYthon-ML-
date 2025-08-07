"""
Unit tests for ResumePDFProcessor class.

Tests PDF processing functionality, metadata extraction, error handling,
and multiple extraction methods.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import os
from pathlib import Path

# Import the classes to test
try:
    from resume_keyword_extractor.processors.pdf_processor import (
        ResumePDFProcessor, 
        PDFExtractionMethod, 
        PDFType, 
        PDFMetadata,
        PDFExtractionResult
    )
except ImportError:
    # Skip tests if dependencies not available
    ResumePDFProcessor = None


@unittest.skipIf(ResumePDFProcessor is None, "PDF processor dependencies not available")
class TestResumePDFProcessor(unittest.TestCase):
    """Test cases for ResumePDFProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        with patch('resume_keyword_extractor.processors.pdf_processor.PYPDF2_AVAILABLE', True):
            with patch('resume_keyword_extractor.processors.pdf_processor.PDFPLUMBER_AVAILABLE', True):
                self.processor = ResumePDFProcessor()
    
    def test_init_with_available_methods(self):
        """Test initialization with available methods."""
        with patch('resume_keyword_extractor.processors.pdf_processor.PYPDF2_AVAILABLE', True):
            with patch('resume_keyword_extractor.processors.pdf_processor.PDFPLUMBER_AVAILABLE', True):
                processor = ResumePDFProcessor()
                self.assertIn(PDFExtractionMethod.PYPDF2, processor.available_methods)
                self.assertIn(PDFExtractionMethod.PDFPLUMBER, processor.available_methods)
    
    def test_init_no_methods_available(self):
        """Test initialization when no methods are available."""
        with patch('resume_keyword_extractor.processors.pdf_processor.PYPDF2_AVAILABLE', False):
            with patch('resume_keyword_extractor.processors.pdf_processor.PDFPLUMBER_AVAILABLE', False):
                with patch('resume_keyword_extractor.processors.pdf_processor.PYMUPDF_AVAILABLE', False):
                    with self.assertRaises(RuntimeError):
                        ResumePDFProcessor()
    
    def test_prepare_pdf_file_streamlit_upload(self):
        """Test PDF file preparation from Streamlit upload."""
        mock_file = Mock()
        mock_file.read.return_value = b"PDF content"
        mock_file.seek = Mock()
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp_file = Mock()
            mock_temp_file.name = '/tmp/test.pdf'
            mock_temp_file.__enter__.return_value = mock_temp_file
            mock_temp.return_value = mock_temp_file
            
            result = self.processor._prepare_pdf_file(mock_file)
            
            self.assertEqual(result, '/tmp/test.pdf')
            mock_file.read.assert_called_once()
            mock_file.seek.assert_called_once_with(0)
    
    def test_prepare_pdf_file_path(self):
        """Test PDF file preparation from file path."""
        file_path = "/path/to/test.pdf"
        result = self.processor._prepare_pdf_file(file_path)
        self.assertEqual(result, file_path)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_analyze_pdf_text_based(self, mock_pypdf2, mock_file):
        """Test PDF analysis for text-based PDF."""
        # Mock PyPDF2 reader
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.metadata = {
            '/Title': 'Test Resume',
            '/Author': 'John Doe',
            '/Creator': 'Test Creator'
        }
        
        # Mock pages with text content
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample resume text content"
        mock_reader.pages = [mock_page]
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        pdf_type, metadata = self.processor._analyze_pdf('/test/path.pdf')
        
        self.assertEqual(pdf_type, PDFType.TEXT_BASED)
        self.assertEqual(metadata.title, 'Test Resume')
        self.assertEqual(metadata.author, 'John Doe')
        self.assertFalse(metadata.is_encrypted)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_analyze_pdf_encrypted(self, mock_pypdf2, mock_file):
        """Test PDF analysis for encrypted PDF."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        pdf_type, metadata = self.processor._analyze_pdf('/test/path.pdf')
        
        self.assertEqual(pdf_type, PDFType.ENCRYPTED)
        self.assertTrue(metadata.is_encrypted)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_analyze_pdf_image_based(self, mock_pypdf2, mock_file):
        """Test PDF analysis for image-based PDF."""
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.metadata = {}
        
        # Mock pages with no text content
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        pdf_type, metadata = self.processor._analyze_pdf('/test/path.pdf')
        
        self.assertEqual(pdf_type, PDFType.IMAGE_BASED)
    
    def test_calculate_text_confidence_empty(self):
        """Test confidence calculation for empty text."""
        confidence = self.processor._calculate_text_confidence("", [])
        self.assertEqual(confidence, 0.0)
    
    def test_calculate_text_confidence_resume_content(self):
        """Test confidence calculation for resume-like content."""
        text = """
        John Doe
        Software Engineer
        
        EXPERIENCE
        Senior Developer at Tech Company
        - Developed web applications using Python and JavaScript
        - Led team of 5 developers
        
        EDUCATION
        Bachelor of Science in Computer Science
        University of Technology
        
        SKILLS
        Python, JavaScript, React, Node.js
        """
        
        page_texts = [text]
        confidence = self.processor._calculate_text_confidence(text, page_texts)
        
        # Should have high confidence due to resume indicators and word count
        self.assertGreater(confidence, 0.7)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_text_confidence_short_text(self):
        """Test confidence calculation for short text."""
        text = "Short text"
        page_texts = [text]
        confidence = self.processor._calculate_text_confidence(text, page_texts)
        
        # Should have lower confidence due to short length
        self.assertLess(confidence, 0.8)
    
    @patch('resume_keyword_extractor.processors.pdf_processor.pdfplumber')
    def test_extract_with_pdfplumber_success(self, mock_pdfplumber):
        """Test successful extraction with pdfplumber."""
        # Mock pdfplumber
        mock_pdf = Mock()
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1: Resume content with experience"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2: Education and skills"
        
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        result = self.processor._extract_with_pdfplumber('/test/path.pdf')
        
        expected_text = "Page 1: Resume content with experience\n\nPage 2: Education and skills"
        self.assertEqual(result.text, expected_text)
        self.assertEqual(result.extraction_method, PDFExtractionMethod.PDFPLUMBER)
        self.assertEqual(len(result.page_texts), 2)
        self.assertGreater(result.confidence, 0.0)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_extract_with_pypdf2_success(self, mock_pypdf2, mock_file):
        """Test successful extraction with PyPDF2."""
        mock_reader = Mock()
        mock_reader.is_encrypted = False
        
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1: Resume content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2: More content"
        
        mock_reader.pages = [mock_page1, mock_page2]
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        result = self.processor._extract_with_pypdf2('/test/path.pdf')
        
        expected_text = "Page 1: Resume content\n\nPage 2: More content"
        self.assertEqual(result.text, expected_text)
        self.assertEqual(result.extraction_method, PDFExtractionMethod.PYPDF2)
        self.assertEqual(len(result.page_texts), 2)
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_extract_with_pypdf2_encrypted_with_password(self, mock_pypdf2, mock_file):
        """Test PyPDF2 extraction with encrypted PDF and correct password."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = True
        
        mock_page = Mock()
        mock_page.extract_text.return_value = "Decrypted content"
        mock_reader.pages = [mock_page]
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        result = self.processor._extract_with_pypdf2('/test/path.pdf', password='correct')
        
        self.assertEqual(result.text, "Decrypted content")
        mock_reader.decrypt.assert_called_once_with('correct')
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_extract_with_pypdf2_encrypted_wrong_password(self, mock_pypdf2, mock_file):
        """Test PyPDF2 extraction with encrypted PDF and wrong password."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = False
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with self.assertRaises(ValueError) as context:
            self.processor._extract_with_pypdf2('/test/path.pdf', password='wrong')
        
        self.assertIn("Invalid password", str(context.exception))
    
    @patch('builtins.open', new_callable=mock_open, read_data=b"PDF content")
    @patch('resume_keyword_extractor.processors.pdf_processor.PyPDF2')
    def test_extract_with_pypdf2_encrypted_no_password(self, mock_pypdf2, mock_file):
        """Test PyPDF2 extraction with encrypted PDF and no password."""
        mock_reader = Mock()
        mock_reader.is_encrypted = True
        
        mock_pypdf2.PdfReader.return_value = mock_reader
        
        with self.assertRaises(ValueError) as context:
            self.processor._extract_with_pypdf2('/test/path.pdf')
        
        self.assertIn("PDF is encrypted", str(context.exception))
    
    def test_get_available_methods(self):
        """Test getting available extraction methods."""
        methods = self.processor.get_available_methods()
        self.assertIsInstance(methods, list)
        self.assertGreater(len(methods), 0)
    
    def test_is_method_available(self):
        """Test checking if specific method is available."""
        # This will depend on what's mocked in setUp
        available = self.processor.is_method_available(PDFExtractionMethod.PYPDF2)
        self.assertIsInstance(available, bool)
    
    def test_pdf_metadata_dataclass(self):
        """Test PDFMetadata dataclass."""
        metadata = PDFMetadata(
            title="Test Resume",
            author="John Doe",
            page_count=2,
            is_encrypted=False
        )
        
        self.assertEqual(metadata.title, "Test Resume")
        self.assertEqual(metadata.author, "John Doe")
        self.assertEqual(metadata.page_count, 2)
        self.assertFalse(metadata.is_encrypted)
    
    def test_pdf_extraction_result_dataclass(self):
        """Test PDFExtractionResult dataclass."""
        metadata = PDFMetadata(page_count=1)
        
        result = PDFExtractionResult(
            text="Sample text",
            confidence=0.95,
            extraction_method=PDFExtractionMethod.PDFPLUMBER,
            pdf_type=PDFType.TEXT_BASED,
            metadata=metadata,
            page_texts=["Sample text"]
        )
        
        self.assertEqual(result.text, "Sample text")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.extraction_method, PDFExtractionMethod.PDFPLUMBER)
        self.assertEqual(result.pdf_type, PDFType.TEXT_BASED)
        self.assertEqual(result.metadata, metadata)
        self.assertEqual(result.page_texts, ["Sample text"])


class TestPDFProcessorBasic(unittest.TestCase):
    """Basic tests that don't require external dependencies."""
    
    def test_pdf_extraction_method_enum(self):
        """Test PDFExtractionMethod enum values."""
        self.assertEqual(PDFExtractionMethod.PDFPLUMBER.value, "pdfplumber")
        self.assertEqual(PDFExtractionMethod.PYPDF2.value, "pypdf2")
        self.assertEqual(PDFExtractionMethod.PYMUPDF.value, "pymupdf")
        self.assertEqual(PDFExtractionMethod.OCR_FALLBACK.value, "ocr_fallback")
    
    def test_pdf_type_enum(self):
        """Test PDFType enum values."""
        self.assertEqual(PDFType.TEXT_BASED.value, "text_based")
        self.assertEqual(PDFType.IMAGE_BASED.value, "image_based")
        self.assertEqual(PDFType.MIXED.value, "mixed")
        self.assertEqual(PDFType.ENCRYPTED.value, "encrypted")
        self.assertEqual(PDFType.CORRUPTED.value, "corrupted")


if __name__ == '__main__':
    unittest.main()