"""
Tests for the ReportGenerator class.

This module contains comprehensive tests for report generation functionality
including PDF, JSON, and CSV export capabilities.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from resume_keyword_extractor.exporters.report_generator import (
    ReportGenerator, ReportConfig, ReportData, GeneratedReport
)
from resume_keyword_extractor.analyzers.base_extractor import (
    SkillMatch, ExtractionResult, SkillCategory, ConfidenceLevel
)
from resume_keyword_extractor.analyzers.skill_categorizer import (
    CategorizationResult, SkillGroup, SkillHierarchy, SkillImportance
)
from resume_keyword_extractor.analyzers.skill_gap_analyzer import (
    GapAnalysisResult, SkillRecommendation, SkillGap, 
    RecommendationPriority, SkillDemandLevel
)
from resume_keyword_extractor.visualizers.chart_generator import VisualizationResult


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    @pytest.fixture
    def report_generator(self):
        """Create a ReportGenerator instance for testing."""
        config = ReportConfig(
            include_visualizations=True,
            include_recommendations=True,
            include_gap_analysis=True,
            max_recommendations=5
        )
        return ReportGenerator(config)
    
    @pytest.fixture
    def sample_extraction_result(self):
        """Create sample extraction result for testing."""
        skills = [
            SkillMatch('Python', SkillCategory.TECHNICAL, 0.9, 'Python development', 0),
            SkillMatch('JavaScript', SkillCategory.TECHNICAL, 0.8, 'JS programming', 50),
            SkillMatch('Communication', SkillCategory.SOFT, 0.7, 'Good communication', 100)
        ]
        
        return ExtractionResult(
            skills=skills,
            total_skills=3,
            categories={SkillCategory.TECHNICAL: 2, SkillCategory.SOFT: 1},
            confidence_distribution={ConfidenceLevel.HIGH: 2, ConfidenceLevel.MEDIUM: 1},
            processing_time=0.5,
            metadata={'test': True}
        )
    
    @pytest.fixture
    def sample_categorization_result(self):
        """Create sample categorization result for testing."""
        skills = [
            SkillMatch('Python', SkillCategory.TECHNICAL, 0.9, 'Python dev', 0),
            SkillMatch('JavaScript', SkillCategory.TECHNICAL, 0.8, 'JS dev', 50)
        ]
        
        skill_group = SkillGroup(
            name='Programming Languages',
            category=SkillCategory.TECHNICAL,
            skills=skills,
            importance=SkillImportance.HIGH
        )
        
        hierarchy = SkillHierarchy(root_category=SkillCategory.TECHNICAL)
        hierarchy.add_group('programming_languages', skill_group)
        
        return CategorizationResult(
            hierarchies={SkillCategory.TECHNICAL: hierarchy},
            skill_groups=[skill_group],
            normalized_skills={'Python': 'Python', 'JavaScript': 'JavaScript'},
            duplicates_removed=0,
            importance_scores={'Python': 0.9, 'JavaScript': 0.8},
            frequency_analysis={'Python': 1, 'JavaScript': 1},
            processing_time=0.3
        )
    
    @pytest.fixture
    def sample_gap_analysis_result(self):
        """Create sample gap analysis result for testing."""
        missing_skills = [
            SkillGap('React', SkillCategory.FRAMEWORK, 0.8, 0.7, ['Angular']),
            SkillGap('SQL', SkillCategory.DATABASE, 0.9, 0.8, [])
        ]
        
        recommendations = [
            SkillRecommendation(
                skill='React',
                category=SkillCategory.FRAMEWORK,
                priority=RecommendationPriority.HIGH,
                market_demand=SkillDemandLevel.VERY_HIGH,
                reason='Essential for frontend development',
                learning_resources=['React Documentation', 'React Tutorial'],
                estimated_learning_time='2-3 months'
            ),
            SkillRecommendation(
                skill='SQL',
                category=SkillCategory.DATABASE,
                priority=RecommendationPriority.CRITICAL,
                market_demand=SkillDemandLevel.HIGH,
                reason='Required for data management',
                learning_resources=['SQL Tutorial', 'W3Schools SQL'],
                estimated_learning_time='1-2 months'
            )
        ]
        
        return GapAnalysisResult(
            target_role='software_developer',
            current_skills=['Python', 'JavaScript'],
            missing_skills=missing_skills,
            recommendations=recommendations,
            skill_match_percentage=65.5,
            strengths=['Python', 'JavaScript'],
            improvement_areas=['Frontend Frameworks', 'Database Skills'],
            career_progression_path=['Junior Developer', 'Developer', 'Senior Developer'],
            processing_time=0.8
        )
    
    @pytest.fixture
    def sample_visualizations(self):
        """Create sample visualizations for testing."""
        return {
            'word_cloud': VisualizationResult(
                chart_type='word_cloud',
                image_data='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',  # 1x1 PNG
                metadata={'total_skills': 5}
            ),
            'skill_distribution': VisualizationResult(
                chart_type='skill_distribution',
                image_data='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
                metadata={'categories': ['Technical', 'Soft']}
            )
        }
    
    @pytest.fixture
    def sample_report_data(self, sample_extraction_result, sample_categorization_result,
                          sample_gap_analysis_result, sample_visualizations):
        """Create complete sample report data."""
        return ReportData(
            extraction_results=[sample_extraction_result],
            categorization_result=sample_categorization_result,
            gap_analysis_result=sample_gap_analysis_result,
            visualizations=sample_visualizations,
            metadata={'test_report': True}
        )
    
    def test_report_generator_initialization(self):
        """Test ReportGenerator initialization."""
        # Test with default config
        generator = ReportGenerator()
        assert generator.config is not None
        assert generator.config.include_visualizations is True
        
        # Test with custom config
        custom_config = ReportConfig(
            include_visualizations=False,
            max_recommendations=3
        )
        generator = ReportGenerator(custom_config)
        assert generator.config.include_visualizations is False
        assert generator.config.max_recommendations == 3
    
    def test_generate_json_report_success(self, report_generator, sample_report_data):
        """Test successful JSON report generation."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = report_generator.generate_json_report(sample_report_data, tmp_path)
            
            assert isinstance(result, GeneratedReport)
            assert result.report_type == 'json'
            assert result.file_path == tmp_path
            assert result.content is not None
            assert result.generation_time > 0
            
            # Verify file was created
            assert os.path.exists(tmp_path)
            
            # Verify JSON content
            with open(tmp_path, 'r') as f:
                json_data = json.load(f)
            
            assert 'report_metadata' in json_data
            assert 'extraction_results' in json_data
            assert 'gap_analysis_result' in json_data
            assert 'summary' in json_data
            
            # Verify extraction results
            assert len(json_data['extraction_results']) == 1
            assert json_data['extraction_results'][0]['total_skills'] == 3
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_generate_json_report_without_file(self, report_generator, sample_report_data):
        """Test JSON report generation without saving to file."""
        result = report_generator.generate_json_report(sample_report_data)
        
        assert isinstance(result, GeneratedReport)
        assert result.report_type == 'json'
        assert result.file_path is None
        assert result.content is not None
        
        # Verify JSON content is valid
        json_data = json.loads(result.content)
        assert 'report_metadata' in json_data
        assert 'extraction_results' in json_data
    
    def test_generate_csv_report_success(self, report_generator, sample_report_data):
        """Test successful CSV report generation."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = report_generator.generate_csv_report(sample_report_data, tmp_path)
            
            assert isinstance(result, GeneratedReport)
            assert result.report_type == 'csv'
            assert result.file_path == tmp_path
            assert result.content is not None
            assert result.generation_time > 0
            
            # Verify file was created
            assert os.path.exists(tmp_path)
            
            # Verify CSV content
            with open(tmp_path, 'r') as f:
                csv_content = f.read()
            
            assert '# SKILLS' in csv_content
            assert 'Python' in csv_content
            assert 'JavaScript' in csv_content
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_generate_csv_report_without_file(self, report_generator, sample_report_data):
        """Test CSV report generation without saving to file."""
        result = report_generator.generate_csv_report(sample_report_data)
        
        assert isinstance(result, GeneratedReport)
        assert result.report_type == 'csv'
        assert result.file_path is None
        assert result.content is not None
        
        # Verify CSV content
        assert '# SKILLS' in result.content
        assert 'Python' in result.content
    
    @patch('resume_keyword_extractor.exporters.report_generator.SimpleDocTemplate')
    def test_generate_pdf_report_success(self, mock_doc, report_generator, sample_report_data):
        """Test successful PDF report generation."""
        # Mock the PDF document
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = report_generator.generate_pdf_report(sample_report_data, tmp_path, "Test Report")
            
            assert isinstance(result, GeneratedReport)
            assert result.report_type == 'pdf'
            assert result.file_path == tmp_path
            assert result.generation_time > 0
            
            # Verify PDF document was created and built
            mock_doc.assert_called_once()
            mock_doc_instance.build.assert_called_once()
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_generate_all_reports_success(self, report_generator, sample_report_data):
        """Test generation of all report formats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, 'test_report')
            
            with patch('resume_keyword_extractor.exporters.report_generator.SimpleDocTemplate'):
                results = report_generator.generate_all_reports(
                    sample_report_data, base_path, "Test Report"
                )
            
            assert len(results) == 3
            assert 'pdf' in results
            assert 'json' in results
            assert 'csv' in results
            
            # Verify all results are GeneratedReport instances
            for report_type, result in results.items():
                assert isinstance(result, GeneratedReport)
                assert result.report_type == report_type
                assert result.generation_time > 0
    
    def test_extraction_result_to_dict(self, report_generator, sample_extraction_result):
        """Test conversion of ExtractionResult to dictionary."""
        result_dict = report_generator._extraction_result_to_dict(sample_extraction_result)
        
        assert isinstance(result_dict, dict)
        assert result_dict['total_skills'] == 3
        assert 'categories' in result_dict
        assert 'skills' in result_dict
        assert len(result_dict['skills']) == 3
        
        # Verify skill structure
        skill = result_dict['skills'][0]
        assert 'skill' in skill
        assert 'category' in skill
        assert 'confidence' in skill
        assert 'context' in skill
    
    def test_categorization_result_to_dict(self, report_generator, sample_categorization_result):
        """Test conversion of CategorizationResult to dictionary."""
        result_dict = report_generator._categorization_result_to_dict(sample_categorization_result)
        
        assert isinstance(result_dict, dict)
        assert 'total_groups' in result_dict
        assert 'skill_groups' in result_dict
        assert 'processing_time' in result_dict
        
        # Verify skill group structure
        assert len(result_dict['skill_groups']) == 1
        group = result_dict['skill_groups'][0]
        assert 'name' in group
        assert 'category' in group
        assert 'skills' in group
    
    def test_gap_analysis_result_to_dict(self, report_generator, sample_gap_analysis_result):
        """Test conversion of GapAnalysisResult to dictionary."""
        result_dict = report_generator._gap_analysis_result_to_dict(sample_gap_analysis_result)
        
        assert isinstance(result_dict, dict)
        assert result_dict['target_role'] == 'software_developer'
        assert 'skill_match_percentage' in result_dict
        assert 'missing_skills' in result_dict
        assert 'recommendations' in result_dict
        
        # Verify missing skills structure
        assert len(result_dict['missing_skills']) == 2
        missing_skill = result_dict['missing_skills'][0]
        assert 'skill' in missing_skill
        assert 'category' in missing_skill
        assert 'importance' in missing_skill
        
        # Verify recommendations structure
        assert len(result_dict['recommendations']) == 2
        recommendation = result_dict['recommendations'][0]
        assert 'skill' in recommendation
        assert 'priority' in recommendation
        assert 'learning_resources' in recommendation
    
    def test_generate_report_summary(self, report_generator, sample_report_data):
        """Test report summary generation."""
        summary = report_generator._generate_report_summary(sample_report_data)
        
        assert isinstance(summary, dict)
        assert 'total_skills_analyzed' in summary
        assert 'analysis_timestamp' in summary
        assert 'target_role' in summary
        assert 'skill_match_percentage' in summary
        
        assert summary['total_skills_analyzed'] == 3
        assert summary['target_role'] == 'software_developer'
        assert summary['skill_match_percentage'] == 65.5
    
    def test_assess_skill_match(self, report_generator):
        """Test skill match assessment."""
        assert report_generator._assess_skill_match(85.0) == "Excellent"
        assert report_generator._assess_skill_match(70.0) == "Good"
        assert report_generator._assess_skill_match(50.0) == "Fair"
        assert report_generator._assess_skill_match(30.0) == "Needs Improvement"
    
    def test_report_config_validation(self):
        """Test ReportConfig validation and defaults."""
        # Test default config
        config = ReportConfig()
        assert config.include_visualizations is True
        assert config.include_recommendations is True
        assert config.max_recommendations == 10
        assert config.font_size == 10
        
        # Test custom config
        custom_config = ReportConfig(
            include_visualizations=False,
            max_recommendations=5,
            font_size=12
        )
        assert custom_config.include_visualizations is False
        assert custom_config.max_recommendations == 5
        assert custom_config.font_size == 12
    
    def test_report_data_structure(self, sample_report_data):
        """Test ReportData structure and content."""
        assert hasattr(sample_report_data, 'extraction_results')
        assert hasattr(sample_report_data, 'categorization_result')
        assert hasattr(sample_report_data, 'gap_analysis_result')
        assert hasattr(sample_report_data, 'visualizations')
        assert hasattr(sample_report_data, 'metadata')
        
        assert len(sample_report_data.extraction_results) == 1
        assert sample_report_data.categorization_result is not None
        assert sample_report_data.gap_analysis_result is not None
        assert sample_report_data.visualizations is not None
    
    def test_empty_report_data_handling(self, report_generator):
        """Test handling of empty report data."""
        empty_data = ReportData(extraction_results=[])
        
        # Should not raise exceptions
        json_result = report_generator.generate_json_report(empty_data)
        assert json_result.report_type == 'json'
        
        csv_result = report_generator.generate_csv_report(empty_data)
        assert csv_result.report_type == 'csv'
        
        # Verify JSON content handles empty data
        json_content = json.loads(json_result.content)
        assert json_content['extraction_results'] == []
        assert json_content['categorization_result'] is None
    
    def test_report_generation_error_handling(self, report_generator, sample_report_data):
        """Test error handling in report generation."""
        # Test with invalid file path
        invalid_path = "/invalid/path/report.json"
        
        with pytest.raises(Exception):
            report_generator.generate_json_report(sample_report_data, invalid_path)
    
    def test_custom_styles_setup(self, report_generator):
        """Test custom styles setup for PDF generation."""
        # Verify custom styles were added
        assert 'CustomTitle' in report_generator.styles
        assert 'CustomHeader' in report_generator.styles
        assert 'CustomSubHeader' in report_generator.styles
        assert 'CustomBody' in report_generator.styles
        
        # Verify style properties
        title_style = report_generator.styles['CustomTitle']
        assert title_style.fontSize == report_generator.config.title_font_size
        
        header_style = report_generator.styles['CustomHeader']
        assert header_style.fontSize == report_generator.config.header_font_size


class TestReportGeneratorIntegration:
    """Integration tests for ReportGenerator with realistic data."""
    
    @pytest.fixture
    def realistic_report_data(self):
        """Create realistic report data for integration testing."""
        # Create realistic skills
        skills = [
            SkillMatch('Python', SkillCategory.TECHNICAL, 0.95, 'Expert Python developer with 5 years experience', 0),
            SkillMatch('JavaScript', SkillCategory.TECHNICAL, 0.88, 'Proficient in modern JavaScript ES6+', 100),
            SkillMatch('React', SkillCategory.FRAMEWORK, 0.82, 'Built multiple React applications', 200),
            SkillMatch('SQL', SkillCategory.DATABASE, 0.75, 'Database design and optimization', 300),
            SkillMatch('Communication', SkillCategory.SOFT, 0.80, 'Strong communication and presentation skills', 400),
            SkillMatch('Leadership', SkillCategory.SOFT, 0.70, 'Led team of 3 developers', 500)
        ]
        
        extraction_result = ExtractionResult(
            skills=skills,
            total_skills=6,
            categories={
                SkillCategory.TECHNICAL: 2,
                SkillCategory.FRAMEWORK: 1,
                SkillCategory.DATABASE: 1,
                SkillCategory.SOFT: 2
            },
            confidence_distribution={
                ConfidenceLevel.HIGH: 4,
                ConfidenceLevel.MEDIUM: 2
            },
            processing_time=1.2
        )
        
        # Create gap analysis
        missing_skills = [
            SkillGap('Docker', SkillCategory.TOOL, 0.85, 0.75),
            SkillGap('AWS', SkillCategory.CLOUD, 0.80, 0.70)
        ]
        
        recommendations = [
            SkillRecommendation(
                skill='Docker',
                category=SkillCategory.TOOL,
                priority=RecommendationPriority.HIGH,
                market_demand=SkillDemandLevel.VERY_HIGH,
                reason='Essential for modern deployment practices',
                learning_resources=['Docker Documentation', 'Docker Mastery Course'],
                estimated_learning_time='1-2 months'
            )
        ]
        
        gap_analysis = GapAnalysisResult(
            target_role='Senior Software Developer',
            current_skills=[s.skill for s in skills],
            missing_skills=missing_skills,
            recommendations=recommendations,
            skill_match_percentage=78.5,
            strengths=['Python', 'JavaScript', 'React'],
            improvement_areas=['DevOps Tools', 'Cloud Platforms'],
            career_progression_path=['Senior Developer', 'Tech Lead', 'Engineering Manager'],
            processing_time=0.9
        )
        
        return ReportData(
            extraction_results=[extraction_result],
            gap_analysis_result=gap_analysis
        )
    
    def test_comprehensive_report_generation(self, realistic_report_data):
        """Test comprehensive report generation with realistic data."""
        generator = ReportGenerator()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, 'comprehensive_report')
            
            # Generate JSON report
            json_result = generator.generate_json_report(realistic_report_data, f"{base_path}.json")
            
            assert json_result.report_type == 'json'
            assert os.path.exists(json_result.file_path)
            
            # Verify JSON content quality
            with open(json_result.file_path, 'r') as f:
                json_data = json.load(f)
            
            assert json_data['summary']['total_skills_analyzed'] == 6
            assert json_data['summary']['target_role'] == 'Senior Software Developer'
            assert json_data['summary']['skill_match_percentage'] == 78.5
            
            # Generate CSV report
            csv_result = generator.generate_csv_report(realistic_report_data, f"{base_path}.csv")
            
            assert csv_result.report_type == 'csv'
            assert os.path.exists(csv_result.file_path)
            
            # Verify CSV content
            with open(csv_result.file_path, 'r') as f:
                csv_content = f.read()
            
            assert 'Python' in csv_content
            assert 'Docker' in csv_content  # Should include recommendations
    
    def test_report_consistency_across_formats(self, realistic_report_data):
        """Test that data is consistent across different report formats."""
        generator = ReportGenerator()
        
        # Generate both JSON and CSV reports
        json_result = generator.generate_json_report(realistic_report_data)
        csv_result = generator.generate_csv_report(realistic_report_data)
        
        # Parse JSON data
        json_data = json.loads(json_result.content)
        
        # Verify consistent skill counts
        json_skills_count = json_data['summary']['total_skills_analyzed']
        
        # Count skills in CSV (approximate check)
        csv_lines = csv_result.content.split('\n')
        csv_skill_lines = [line for line in csv_lines if 'Python' in line or 'JavaScript' in line]
        
        assert json_skills_count == 6
        assert len(csv_skill_lines) > 0  # Should have skill data in CSV
    
    def test_large_dataset_performance(self):
        """Test report generation performance with larger datasets."""
        # Create large dataset
        skills = []
        for i in range(100):
            skills.append(SkillMatch(
                skill=f'Skill_{i}',
                category=SkillCategory.TECHNICAL,
                confidence=0.5 + (i % 50) / 100,
                context=f'Context for skill {i}',
                position=i * 10
            ))
        
        extraction_result = ExtractionResult(
            skills=skills,
            total_skills=100,
            categories={SkillCategory.TECHNICAL: 100},
            confidence_distribution={ConfidenceLevel.MEDIUM: 100},
            processing_time=2.0
        )
        
        large_data = ReportData(extraction_results=[extraction_result])
        generator = ReportGenerator()
        
        # Test JSON generation performance
        import time
        start_time = time.time()
        json_result = generator.generate_json_report(large_data)
        json_time = time.time() - start_time
        
        assert json_result.report_type == 'json'
        assert json_time < 5.0  # Should complete within 5 seconds
        
        # Verify data integrity
        json_data = json.loads(json_result.content)
        assert json_data['summary']['total_skills_analyzed'] == 100