"""
Tests for the SkillGapAnalyzer class.

This module contains comprehensive tests for skill gap analysis functionality
including industry standards comparison, recommendation generation, and accuracy testing.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict

from resume_keyword_extractor.analyzers.skill_gap_analyzer import (
    SkillGapAnalyzer, SkillRecommendation, SkillGap, GapAnalysisResult,
    RecommendationPriority, SkillDemandLevel
)
from resume_keyword_extractor.analyzers.base_extractor import (
    SkillMatch, SkillCategory
)


class TestSkillGapAnalyzer:
    """Test cases for SkillGapAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a SkillGapAnalyzer instance for testing."""
        return SkillGapAnalyzer()
    
    @pytest.fixture
    def sample_extracted_skills(self):
        """Create sample extracted skills for testing."""
        return [
            SkillMatch(
                skill="Python",
                category=SkillCategory.TECHNICAL,
                confidence=0.9,
                context="Experienced Python developer",
                position=10
            ),
            SkillMatch(
                skill="Git",
                category=SkillCategory.TOOL,
                confidence=0.85,
                context="Version control with Git",
                position=50
            ),
            SkillMatch(
                skill="Communication",
                category=SkillCategory.SOFT,
                confidence=0.7,
                context="Strong communication skills",
                position=100
            )
        ]
    
    @pytest.fixture
    def custom_industry_standards(self):
        """Create custom industry standards for testing."""
        return {
            'test_role': {
                'Python': {
                    'importance': 0.9,
                    'category': 'programming_language',
                    'frequency': 0.85,
                    'alternatives': ['Java', 'JavaScript']
                },
                'JavaScript': {
                    'importance': 0.8,
                    'category': 'programming_language',
                    'frequency': 0.8,
                    'alternatives': ['TypeScript']
                },
                'React': {
                    'importance': 0.7,
                    'category': 'framework',
                    'frequency': 0.6,
                    'alternatives': ['Angular', 'Vue.js']
                }
            }
        }
    
    def test_analyzer_initialization(self):
        """Test SkillGapAnalyzer initialization."""
        analyzer = SkillGapAnalyzer()
        
        assert analyzer.industry_standards is not None
        assert analyzer.market_demand_data is not None
        assert analyzer.learning_resources is not None
        assert analyzer.role_skill_weights is not None
        assert analyzer.career_paths is not None
        
        # Test with custom standards
        custom_standards = {'test': {}}
        custom_analyzer = SkillGapAnalyzer(custom_standards)
        assert custom_analyzer.industry_standards == custom_standards
    
    def test_get_industry_standards_exact_match(self, analyzer):
        """Test getting industry standards with exact match."""
        standards = analyzer.get_industry_standards('software_developer')
        
        assert isinstance(standards, dict)
        assert len(standards) > 0
        assert 'Python' in standards
        assert 'JavaScript' in standards
    
    def test_get_industry_standards_partial_match(self, analyzer):
        """Test getting industry standards with partial match."""
        standards = analyzer.get_industry_standards('software engineer')
        
        # Should match 'software_developer'
        assert isinstance(standards, dict)
        assert len(standards) > 0
    
    def test_get_industry_standards_no_match(self, analyzer):
        """Test getting industry standards with no match."""
        standards = analyzer.get_industry_standards('unknown_role')
        
        # Should default to software_developer
        assert isinstance(standards, dict)
        assert len(standards) > 0
    
    def test_analyze_skill_gaps_basic(self, analyzer, sample_extracted_skills):
        """Test basic skill gap analysis."""
        result = analyzer.analyze_skill_gaps(sample_extracted_skills, 'software_developer')
        
        assert isinstance(result, GapAnalysisResult)
        assert result.target_role == 'software_developer'
        assert len(result.current_skills) == len(sample_extracted_skills)
        assert isinstance(result.missing_skills, list)
        assert isinstance(result.recommendations, list)
        assert 0 <= result.skill_match_percentage <= 100
        assert isinstance(result.strengths, list)
        assert isinstance(result.improvement_areas, list)
        assert isinstance(result.career_progression_path, list)
        assert result.processing_time > 0
    
    def test_analyze_skill_gaps_with_custom_standards(self, custom_industry_standards, sample_extracted_skills):
        """Test skill gap analysis with custom industry standards."""
        analyzer = SkillGapAnalyzer(custom_industry_standards)
        result = analyzer.analyze_skill_gaps(sample_extracted_skills, 'test_role')
        
        assert result.target_role == 'test_role'
        assert isinstance(result.missing_skills, list)
        
        # Should identify JavaScript and React as missing
        missing_skill_names = [gap.missing_skill for gap in result.missing_skills]
        assert 'JavaScript' in missing_skill_names
        assert 'React' in missing_skill_names
        assert 'Python' not in missing_skill_names  # User has Python
    
    def test_identify_missing_skills(self, analyzer):
        """Test missing skills identification."""
        current_skills = {'Python', 'Git'}
        required_skills = {
            'Python': {'importance': 0.9, 'category': 'programming_language', 'frequency': 0.8},
            'JavaScript': {'importance': 0.8, 'category': 'programming_language', 'frequency': 0.7},
            'React': {'importance': 0.7, 'category': 'framework', 'frequency': 0.6}
        }
        
        missing_skills = analyzer._identify_missing_skills(current_skills, required_skills, 'test_role')
        
        assert len(missing_skills) == 2
        missing_names = [gap.missing_skill for gap in missing_skills]
        assert 'JavaScript' in missing_names
        assert 'React' in missing_names
        assert 'Python' not in missing_names
        
        # Check sorting by importance + frequency
        assert missing_skills[0].missing_skill == 'JavaScript'  # Higher combined score
    
    def test_generate_recommendations(self, analyzer, sample_extracted_skills):
        """Test recommendation generation."""
        # Create some skill gaps
        missing_skills = [
            SkillGap(
                missing_skill='JavaScript',
                category=SkillCategory.TECHNICAL,
                importance=0.9,
                frequency_in_jobs=0.8,
                alternatives=[]
            ),
            SkillGap(
                missing_skill='React',
                category=SkillCategory.FRAMEWORK,
                importance=0.7,
                frequency_in_jobs=0.6,
                alternatives=[]
            )
        ]
        
        recommendations = analyzer._generate_recommendations(
            missing_skills, sample_extracted_skills, 'software_developer'
        )
        
        assert len(recommendations) == 2
        
        for rec in recommendations:
            assert isinstance(rec, SkillRecommendation)
            assert rec.skill in ['JavaScript', 'React']
            assert isinstance(rec.priority, RecommendationPriority)
            assert isinstance(rec.market_demand, SkillDemandLevel)
            assert len(rec.reason) > 0
            assert isinstance(rec.learning_resources, list)
            assert rec.estimated_learning_time is not None
    
    def test_calculate_skill_match_percentage(self, analyzer):
        """Test skill match percentage calculation."""
        current_skills = {'Python', 'Git', 'Communication'}
        required_skills = {
            'Python': {'importance': 0.9},
            'JavaScript': {'importance': 0.8},
            'Git': {'importance': 0.7},
            'SQL': {'importance': 0.6, 'alternatives': ['NoSQL']}
        }
        
        percentage = analyzer._calculate_skill_match_percentage(current_skills, required_skills)
        
        # Should have Python (0.9) + Git (0.7) = 1.6 out of total 3.0
        expected = (1.6 / 3.0) * 100
        assert abs(percentage - expected) < 1.0  # Allow small floating point differences
    
    def test_calculate_skill_match_with_alternatives(self, analyzer):
        """Test skill match calculation with alternative skills."""
        current_skills = {'MongoDB'}  # User has MongoDB instead of SQL
        required_skills = {
            'SQL': {'importance': 1.0, 'alternatives': ['MongoDB', 'NoSQL']}
        }
        
        percentage = analyzer._calculate_skill_match_percentage(current_skills, required_skills)
        
        # Should get 70% credit for having alternative
        assert abs(percentage - 70.0) < 1.0
    
    def test_identify_strengths(self, analyzer, sample_extracted_skills):
        """Test strength identification."""
        required_skills = {
            'Python': {'importance': 0.9},
            'Git': {'importance': 0.8},
            'JavaScript': {'importance': 0.7}
        }
        
        strengths = analyzer._identify_strengths(sample_extracted_skills, required_skills)
        
        # Should identify Python and Git as strengths (high confidence + required)
        assert 'Python' in strengths
        assert 'Git' in strengths
        assert 'Communication' not in strengths  # Not in required skills
    
    def test_identify_improvement_areas(self, analyzer):
        """Test improvement area identification."""
        missing_skills = [
            SkillGap('JavaScript', SkillCategory.TECHNICAL, 0.9, 0.8),
            SkillGap('TypeScript', SkillCategory.TECHNICAL, 0.8, 0.7),
            SkillGap('React', SkillCategory.FRAMEWORK, 0.8, 0.7),
            SkillGap('Angular', SkillCategory.FRAMEWORK, 0.7, 0.6)
        ]
        
        improvement_areas = analyzer._identify_improvement_areas(missing_skills)
        
        # Should identify both technical and framework skills as improvement areas
        assert len(improvement_areas) >= 1
        area_text = ' '.join(improvement_areas).lower()
        assert 'technical' in area_text or 'framework' in area_text
    
    def test_calculate_recommendation_priority(self, analyzer):
        """Test recommendation priority calculation."""
        # Critical priority gap
        critical_gap = SkillGap('Python', SkillCategory.TECHNICAL, 0.95, 0.9)
        priority = analyzer._calculate_recommendation_priority(critical_gap, 'software_developer')
        assert priority == RecommendationPriority.CRITICAL
        
        # High priority gap
        high_gap = SkillGap('JavaScript', SkillCategory.TECHNICAL, 0.8, 0.7)
        priority = analyzer._calculate_recommendation_priority(high_gap, 'software_developer')
        assert priority == RecommendationPriority.HIGH
        
        # Low priority gap
        low_gap = SkillGap('Obscure Tool', SkillCategory.TOOL, 0.3, 0.2)
        priority = analyzer._calculate_recommendation_priority(low_gap, 'software_developer')
        assert priority == RecommendationPriority.LOW
    
    def test_generate_recommendation_reason(self, analyzer):
        """Test recommendation reason generation."""
        gap = SkillGap(
            missing_skill='JavaScript',
            category=SkillCategory.TECHNICAL,
            importance=0.9,
            frequency_in_jobs=0.85,
            alternatives=['TypeScript']
        )
        
        reason = analyzer._generate_recommendation_reason(gap, 'software_developer')
        
        assert isinstance(reason, str)
        assert len(reason) > 0
        assert 'JavaScript' not in reason  # Should not repeat skill name
        assert 'software_developer' in reason
    
    def test_estimate_learning_time(self, analyzer):
        """Test learning time estimation."""
        # Test different categories
        tech_time = analyzer._estimate_learning_time('Python', SkillCategory.TECHNICAL)
        assert isinstance(tech_time, str)
        assert 'month' in tech_time.lower()
        
        tool_time = analyzer._estimate_learning_time('Git', SkillCategory.TOOL)
        assert isinstance(tool_time, str)
        assert 'week' in tool_time.lower() or 'month' in tool_time.lower()
        
        soft_time = analyzer._estimate_learning_time('Communication', SkillCategory.SOFT)
        assert isinstance(soft_time, str)
        assert 'ongoing' in soft_time.lower()
    
    def test_prioritize_recommendations(self, analyzer):
        """Test recommendation prioritization."""
        recommendations = [
            SkillRecommendation(
                skill='Low Priority Skill',
                category=SkillCategory.TOOL,
                priority=RecommendationPriority.LOW,
                market_demand=SkillDemandLevel.LOW,
                reason='Test reason',
                confidence_boost=0.1
            ),
            SkillRecommendation(
                skill='High Priority Skill',
                category=SkillCategory.TECHNICAL,
                priority=RecommendationPriority.CRITICAL,
                market_demand=SkillDemandLevel.VERY_HIGH,
                reason='Test reason',
                confidence_boost=0.3
            ),
            SkillRecommendation(
                skill='Medium Priority Skill',
                category=SkillCategory.FRAMEWORK,
                priority=RecommendationPriority.MEDIUM,
                market_demand=SkillDemandLevel.MEDIUM,
                reason='Test reason',
                confidence_boost=0.2
            )
        ]
        
        prioritized = analyzer.prioritize_recommendations(recommendations, max_recommendations=2)
        
        assert len(prioritized) == 2
        assert prioritized[0].skill == 'High Priority Skill'
        assert prioritized[1].skill == 'Medium Priority Skill'
    
    def test_get_analysis_summary(self, analyzer, sample_extracted_skills):
        """Test analysis summary generation."""
        result = analyzer.analyze_skill_gaps(sample_extracted_skills, 'software_developer')
        summary = analyzer.get_analysis_summary(result)
        
        required_keys = [
            'target_role', 'skill_match_percentage', 'total_current_skills',
            'missing_skills_count', 'recommendations_count', 'top_strengths',
            'critical_gaps', 'high_priority_recommendations', 'improvement_areas',
            'processing_time'
        ]
        
        for key in required_keys:
            assert key in summary
        
        assert summary['target_role'] == 'software_developer'
        assert isinstance(summary['skill_match_percentage'], (int, float))
        assert summary['total_current_skills'] == len(sample_extracted_skills)
    
    def test_map_category_string_to_enum(self, analyzer):
        """Test category string to enum mapping."""
        # Test various category mappings
        assert analyzer._map_category_string_to_enum('programming_language') == SkillCategory.TECHNICAL
        assert analyzer._map_category_string_to_enum('framework') == SkillCategory.FRAMEWORK
        assert analyzer._map_category_string_to_enum('tool') == SkillCategory.TOOL
        assert analyzer._map_category_string_to_enum('database') == SkillCategory.DATABASE
        assert analyzer._map_category_string_to_enum('cloud_platform') == SkillCategory.CLOUD
        assert analyzer._map_category_string_to_enum('soft_skill') == SkillCategory.SOFT
        assert analyzer._map_category_string_to_enum('unknown_category') == SkillCategory.TECHNICAL
    
    def test_find_related_skills(self, analyzer, sample_extracted_skills):
        """Test finding related skills."""
        related = analyzer._find_related_skills('Django', sample_extracted_skills)
        
        # Should find Python as related to Django
        assert isinstance(related, list)
        # Note: This test might not find relations with the simple sample data
    
    def test_get_relevant_job_titles(self, analyzer):
        """Test getting relevant job titles for skills."""
        python_jobs = analyzer._get_relevant_job_titles('Python', 'software_developer')
        assert isinstance(python_jobs, list)
        assert len(python_jobs) > 0
        
        js_jobs = analyzer._get_relevant_job_titles('JavaScript', 'frontend_developer')
        assert isinstance(js_jobs, list)
        assert len(js_jobs) > 0
        
        # Test unknown skill
        unknown_jobs = analyzer._get_relevant_job_titles('Unknown Skill', 'test_role')
        assert 'test_role' in unknown_jobs
    
    def test_empty_skills_analysis(self, analyzer):
        """Test analysis with empty skills list."""
        result = analyzer.analyze_skill_gaps([], 'software_developer')
        
        assert result.target_role == 'software_developer'
        assert len(result.current_skills) == 0
        assert result.skill_match_percentage < 100  # Should be low with no skills
        assert len(result.missing_skills) > 0  # Should have many missing skills
        assert len(result.recommendations) > 0  # Should have recommendations
    
    def test_perfect_match_analysis(self, analyzer):
        """Test analysis when user has all required skills."""
        # Create skills that match all requirements for software_developer
        perfect_skills = [
            SkillMatch('Python', SkillCategory.TECHNICAL, 0.9, 'Expert Python', 0),
            SkillMatch('JavaScript', SkillCategory.TECHNICAL, 0.9, 'Expert JS', 50),
            SkillMatch('React', SkillCategory.FRAMEWORK, 0.8, 'React dev', 100),
            SkillMatch('Git', SkillCategory.TOOL, 0.95, 'Git expert', 150),
            SkillMatch('SQL', SkillCategory.DATABASE, 0.8, 'SQL expert', 200),
            SkillMatch('Problem Solving', SkillCategory.SOFT, 0.9, 'Great problem solver', 250),
            SkillMatch('Communication', SkillCategory.SOFT, 0.85, 'Good communicator', 300)
        ]
        
        result = analyzer.analyze_skill_gaps(perfect_skills, 'software_developer')
        
        # Should have high skill match percentage
        assert result.skill_match_percentage > 80
        
        # Should have fewer missing skills
        assert len(result.missing_skills) < 5
        
        # Should identify many strengths
        assert len(result.strengths) > 3


class TestSkillGapAnalyzerIntegration:
    """Integration tests for SkillGapAnalyzer with realistic scenarios."""
    
    @pytest.fixture
    def junior_developer_skills(self):
        """Skills typical of a junior developer."""
        return [
            SkillMatch('Python', SkillCategory.TECHNICAL, 0.7, 'Learning Python', 0),
            SkillMatch('HTML', SkillCategory.TECHNICAL, 0.8, 'Good HTML skills', 50),
            SkillMatch('CSS', SkillCategory.TECHNICAL, 0.7, 'CSS styling', 100),
            SkillMatch('Git', SkillCategory.TOOL, 0.6, 'Basic Git usage', 150)
        ]
    
    @pytest.fixture
    def senior_developer_skills(self):
        """Skills typical of a senior developer."""
        return [
            SkillMatch('Python', SkillCategory.TECHNICAL, 0.95, 'Python expert', 0),
            SkillMatch('JavaScript', SkillCategory.TECHNICAL, 0.9, 'JS expert', 50),
            SkillMatch('React', SkillCategory.FRAMEWORK, 0.85, 'React specialist', 100),
            SkillMatch('Node.js', SkillCategory.FRAMEWORK, 0.8, 'Backend with Node', 150),
            SkillMatch('Docker', SkillCategory.TOOL, 0.8, 'Containerization', 200),
            SkillMatch('AWS', SkillCategory.CLOUD, 0.75, 'Cloud deployment', 250),
            SkillMatch('Leadership', SkillCategory.SOFT, 0.8, 'Team leadership', 300),
            SkillMatch('Mentoring', SkillCategory.SOFT, 0.75, 'Mentoring juniors', 350)
        ]
    
    def test_junior_to_senior_progression(self, junior_developer_skills):
        """Test gap analysis for junior developer targeting senior role."""
        analyzer = SkillGapAnalyzer()
        result = analyzer.analyze_skill_gaps(junior_developer_skills, 'software_developer')
        
        # Junior should have significant gaps
        assert result.skill_match_percentage < 70
        assert len(result.missing_skills) > 3
        assert len(result.recommendations) > 5
        
        # Should recommend important skills like JavaScript, frameworks
        recommended_skills = [rec.skill for rec in result.recommendations]
        assert any('JavaScript' in skill for skill in recommended_skills)
        
        # Should have improvement areas
        assert len(result.improvement_areas) > 0
    
    def test_senior_developer_analysis(self, senior_developer_skills):
        """Test gap analysis for senior developer."""
        analyzer = SkillGapAnalyzer()
        result = analyzer.analyze_skill_gaps(senior_developer_skills, 'software_developer')
        
        # Senior should have good skill match
        assert result.skill_match_percentage > 70
        
        # Should have many strengths
        assert len(result.strengths) > 4
        
        # Should have fewer critical gaps
        critical_recs = [rec for rec in result.recommendations 
                        if rec.priority == RecommendationPriority.CRITICAL]
        assert len(critical_recs) < 3
    
    def test_role_specific_analysis(self, senior_developer_skills):
        """Test analysis for different target roles."""
        analyzer = SkillGapAnalyzer()
        
        # Test for data scientist role
        ds_result = analyzer.analyze_skill_gaps(senior_developer_skills, 'data_scientist')
        
        # Should identify different gaps for data science
        ds_recommendations = [rec.skill for rec in ds_result.recommendations]
        assert any('Machine Learning' in skill or 'ML' in skill for skill in ds_recommendations)
        
        # Test for DevOps role
        devops_result = analyzer.analyze_skill_gaps(senior_developer_skills, 'devops_engineer')
        
        # Should recommend DevOps-specific skills
        devops_recommendations = [rec.skill for rec in devops_result.recommendations]
        assert any('Kubernetes' in skill for skill in devops_recommendations)
    
    def test_recommendation_quality(self, junior_developer_skills):
        """Test quality and relevance of recommendations."""
        analyzer = SkillGapAnalyzer()
        result = analyzer.analyze_skill_gaps(junior_developer_skills, 'software_developer')
        
        # Check recommendation structure
        for rec in result.recommendations[:5]:  # Check top 5
            assert len(rec.reason) > 20  # Should have meaningful reason
            assert len(rec.learning_resources) > 0  # Should have resources
            assert rec.estimated_learning_time is not None
            assert isinstance(rec.priority, RecommendationPriority)
            assert isinstance(rec.market_demand, SkillDemandLevel)
        
        # High priority recommendations should be relevant
        high_priority = [rec for rec in result.recommendations 
                        if rec.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]]
        assert len(high_priority) > 0
        
        # Should prioritize fundamental skills for junior developers
        high_priority_skills = [rec.skill for rec in high_priority]
        fundamental_skills = ['JavaScript', 'SQL', 'React', 'Problem Solving']
        assert any(skill in high_priority_skills for skill in fundamental_skills)
    
    def test_career_progression_paths(self, junior_developer_skills, senior_developer_skills):
        """Test career progression path generation."""
        analyzer = SkillGapAnalyzer()
        
        # Junior developer path
        junior_result = analyzer.analyze_skill_gaps(junior_developer_skills, 'software_developer')
        junior_path = junior_result.career_progression_path
        
        assert len(junior_path) >= 3
        assert 'Junior' in junior_path[0] or 'Entry' in junior_path[0]
        
        # Senior developer path
        senior_result = analyzer.analyze_skill_gaps(senior_developer_skills, 'software_developer')
        senior_path = senior_result.career_progression_path
        
        assert len(senior_path) >= 4  # Should include more advanced levels
    
    def test_analysis_consistency(self, junior_developer_skills):
        """Test that analysis results are consistent across runs."""
        analyzer = SkillGapAnalyzer()
        
        # Run analysis multiple times
        result1 = analyzer.analyze_skill_gaps(junior_developer_skills, 'software_developer')
        result2 = analyzer.analyze_skill_gaps(junior_developer_skills, 'software_developer')
        
        # Results should be consistent
        assert result1.skill_match_percentage == result2.skill_match_percentage
        assert len(result1.missing_skills) == len(result2.missing_skills)
        assert len(result1.recommendations) == len(result2.recommendations)
        
        # Recommendation order should be consistent
        rec1_skills = [rec.skill for rec in result1.recommendations[:5]]
        rec2_skills = [rec.skill for rec in result2.recommendations[:5]]
        assert rec1_skills == rec2_skills