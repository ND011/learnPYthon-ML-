"""
Unit tests for TechnicalSkillsAnalyzer class.

Tests technical skill extraction, categorization, pattern matching,
and context-aware analysis functionality.
"""

import unittest
from unittest.mock import Mock, patch
import time

# Import the classes to test
from resume_keyword_extractor.analyzers.technical_analyzer import TechnicalSkillsAnalyzer
from resume_keyword_extractor.analyzers.base_extractor import (
    SkillMatch, ExtractionResult, SkillCategory, ConfidenceLevel
)


class TestTechnicalSkillsAnalyzer(unittest.TestCase):
    """Test cases for TechnicalSkillsAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock technical skills database
        self.mock_database = {
            "Python": {
                "category": "programming_languages",
                "variations": ["python3", "py"],
                "synonyms": ["python programming"],
                "domain": "Programming Languages",
                "popularity": 0.9
            },
            "JavaScript": {
                "category": "programming_languages",
                "variations": ["js", "javascript"],
                "synonyms": ["js programming"],
                "domain": "Programming Languages",
                "popularity": 0.95
            },
            "React": {
                "category": "frameworks",
                "variations": ["reactjs", "react.js"],
                "synonyms": ["react framework"],
                "domain": "Web Frameworks",
                "popularity": 0.85
            },
            "MySQL": {
                "category": "databases",
                "variations": ["mysql"],
                "synonyms": ["mysql database"],
                "domain": "Databases",
                "popularity": 0.8
            },
            "AWS": {
                "category": "cloud",
                "variations": ["amazon web services"],
                "synonyms": ["amazon cloud"],
                "domain": "Cloud Platforms",
                "popularity": 0.9
            },
            "Docker": {
                "category": "tools",
                "variations": ["docker"],
                "synonyms": ["containerization"],
                "domain": "DevOps Tools",
                "popularity": 0.8
            }
        }
        
        self.analyzer = TechnicalSkillsAnalyzer(self.mock_database)
    
    def test_initialization(self):
        """Test TechnicalSkillsAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer.logger)
        self.assertEqual(self.analyzer.skill_database, self.mock_database)
        self.assertIn('python_versions', self.analyzer.technical_patterns)
        self.assertIn('experience_indicators', self.analyzer.technical_context_patterns)
    
    def test_get_supported_categories(self):
        """Test getting supported skill categories."""
        categories = self.analyzer.get_supported_categories()
        
        expected_categories = [
            SkillCategory.TECHNICAL,
            SkillCategory.FRAMEWORK,
            SkillCategory.TOOL,
            SkillCategory.DATABASE,
            SkillCategory.CLOUD,
            SkillCategory.METHODOLOGY
        ]
        
        for category in expected_categories:
            self.assertIn(category, categories)
    
    def test_preprocess_text(self):
        """Test text preprocessing for technical skills."""
        text = "I know JS, TS, and ML. Also worked with CI/CD and K8s."
        processed = self.analyzer._preprocess_text(text)
        
        self.assertIn("JavaScript", processed)
        self.assertIn("TypeScript", processed)
        self.assertIn("Machine Learning", processed)
        self.assertIn("CI CD", processed)
        self.assertIn("Kubernetes", processed)
    
    def test_determine_skill_category(self):
        """Test skill category determination."""
        # Test with database entry
        python_data = self.mock_database["Python"]
        category = self.analyzer._determine_skill_category("Python", python_data)
        self.assertEqual(category, SkillCategory.TECHNICAL)
        
        react_data = self.mock_database["React"]
        category = self.analyzer._determine_skill_category("React", react_data)
        self.assertEqual(category, SkillCategory.FRAMEWORK)
        
        # Test with unknown skill (fallback logic)
        category = self.analyzer._determine_skill_category("UnknownSkill", {})
        self.assertEqual(category, SkillCategory.TECHNICAL)
        
        # Test pattern-based categorization
        category = self.analyzer._determine_skill_category("MongoDB", {})
        self.assertEqual(category, SkillCategory.DATABASE)
    
    def test_determine_category_from_pattern(self):
        """Test category determination from pattern names."""
        self.assertEqual(
            self.analyzer._determine_category_from_pattern('python_versions'),
            SkillCategory.TECHNICAL
        )
        self.assertEqual(
            self.analyzer._determine_category_from_pattern('react_ecosystem'),
            SkillCategory.FRAMEWORK
        )
        self.assertEqual(
            self.analyzer._determine_category_from_pattern('sql_variants'),
            SkillCategory.DATABASE
        )
        self.assertEqual(
            self.analyzer._determine_category_from_pattern('aws_services'),
            SkillCategory.CLOUD
        )
    
    def test_map_pattern_to_skill(self):
        """Test mapping regex patterns to skill names."""
        self.assertEqual(
            self.analyzer._map_pattern_to_skill('python_versions', 'Python 3.8'),
            'Python'
        )
        self.assertEqual(
            self.analyzer._map_pattern_to_skill('javascript_variants', 'JavaScript ES6'),
            'JavaScript'
        )
        self.assertEqual(
            self.analyzer._map_pattern_to_skill('react_ecosystem', 'React v17'),
            'React'
        )
    
    def test_extract_vcs_tool(self):
        """Test version control tool extraction."""
        self.assertEqual(self.analyzer._extract_vcs_tool('GitHub Actions'), 'GitHub')
        self.assertEqual(self.analyzer._extract_vcs_tool('GitLab CI'), 'GitLab')
        self.assertEqual(self.analyzer._extract_vcs_tool('Bitbucket'), 'Bitbucket')
        self.assertEqual(self.analyzer._extract_vcs_tool('Git commands'), 'Git')
    
    def test_extract_container_tool(self):
        """Test containerization tool extraction."""
        self.assertEqual(self.analyzer._extract_container_tool('Kubernetes cluster'), 'Kubernetes')
        self.assertEqual(self.analyzer._extract_container_tool('K8s deployment'), 'Kubernetes')
        self.assertEqual(self.analyzer._extract_container_tool('Docker containers'), 'Docker')
        self.assertEqual(self.analyzer._extract_container_tool('Podman'), 'Podman')
    
    def test_identify_technical_sections(self):
        """Test identification of technical sections in resume."""
        resume_text = """
        John Doe
        Software Engineer
        
        EXPERIENCE
        Senior Developer at Tech Corp
        
        TECHNICAL SKILLS
        Programming Languages: Python, JavaScript, Java
        Frameworks: React, Django, Spring Boot
        Databases: MySQL, PostgreSQL, MongoDB
        
        EDUCATION
        Bachelor of Computer Science
        """
        
        sections = self.analyzer._identify_technical_sections(resume_text)
        
        self.assertGreater(len(sections), 0)
        
        # Check that technical skills section was found
        found_tech_section = False
        for start, end, section_type in sections:
            if section_type == 'technical_skills':
                section_text = resume_text[start:end]
                self.assertIn('Programming Languages', section_text)
                found_tech_section = True
        
        self.assertTrue(found_tech_section)
    
    def test_extract_years_experience(self):
        """Test extraction of years of experience from context."""
        context1 = "5 years of experience with Python programming"
        self.assertEqual(self.analyzer._extract_years_experience(context1), 5)
        
        context2 = "3 yrs experience in JavaScript development"
        self.assertEqual(self.analyzer._extract_years_experience(context2), 3)
        
        context3 = "Experience with React framework"
        self.assertIsNone(self.analyzer._extract_years_experience(context3))
    
    def test_extract_proficiency_level(self):
        """Test extraction of proficiency level from context."""
        context1 = "Expert in Python programming"
        self.assertEqual(self.analyzer._extract_proficiency_level(context1), "Expert")
        
        context2 = "Advanced knowledge of JavaScript"
        self.assertEqual(self.analyzer._extract_proficiency_level(context2), "Advanced")
        
        context3 = "Intermediate skills in React"
        self.assertEqual(self.analyzer._extract_proficiency_level(context3), "Intermediate")
        
        context4 = "Used Python for projects"
        self.assertIsNone(self.analyzer._extract_proficiency_level(context4))
    
    def test_calculate_technical_confidence(self):
        """Test technical confidence calculation."""
        skill_data = self.mock_database["Python"]
        
        # High confidence context
        high_context = "5 years of experience developing Python applications with Django framework"
        confidence = self.analyzer._calculate_technical_confidence(
            "Python", high_context, "exact", skill_data
        )
        self.assertGreater(confidence, 0.8)
        
        # Medium confidence context
        medium_context = "Used Python for data analysis projects"
        confidence = self.analyzer._calculate_technical_confidence(
            "Python", medium_context, "exact", skill_data
        )
        self.assertGreater(confidence, 0.6)
        self.assertLess(confidence, 0.9)
        
        # Low confidence context
        low_context = "Mentioned Python briefly"
        confidence = self.analyzer._calculate_technical_confidence(
            "Python", low_context, "fuzzy", skill_data
        )
        self.assertLess(confidence, 0.7)
    
    def test_extract_exact_matches(self):
        """Test exact skill matching from database."""
        text = "I have experience with Python programming and React development."
        
        matches = self.analyzer._extract_exact_matches(text)
        
        # Should find Python and React
        skill_names = [match.skill for match in matches]
        self.assertIn("Python", skill_names)
        self.assertIn("React", skill_names)
        
        # Check categories
        for match in matches:
            if match.skill == "Python":
                self.assertEqual(match.category, SkillCategory.TECHNICAL)
            elif match.skill == "React":
                self.assertEqual(match.category, SkillCategory.FRAMEWORK)
    
    def test_extract_skills_comprehensive(self):
        """Test comprehensive skill extraction."""
        resume_text = """
        John Doe - Senior Software Engineer
        
        TECHNICAL SKILLS
        Programming Languages: Python (5 years), JavaScript (ES6), Java
        Web Frameworks: React, Django, Spring Boot
        Databases: MySQL, PostgreSQL, MongoDB
        Cloud Platforms: AWS (EC2, S3, Lambda), Azure
        Tools: Docker, Kubernetes, Git, Jenkins
        
        EXPERIENCE
        Senior Developer at Tech Corp (2019-2023)
        - Developed scalable web applications using Python and Django
        - Built responsive frontend interfaces with React and JavaScript
        - Implemented CI/CD pipelines using Jenkins and Docker
        - Managed cloud infrastructure on AWS including EC2 and RDS
        - Expert in database design with PostgreSQL and MySQL
        
        PROJECTS
        E-commerce Platform
        - Built using React, Node.js, and MongoDB
        - Deployed on AWS with Docker containers
        """
        
        result = self.analyzer.extract_skills(resume_text)
        
        # Check that skills were found
        self.assertGreater(result.total_skills, 0)
        self.assertIsInstance(result, ExtractionResult)
        
        # Check for expected skills
        skill_names = [skill.skill for skill in result.skills]
        expected_skills = ["Python", "JavaScript", "React", "Django", "MySQL", "AWS", "Docker"]
        
        for expected_skill in expected_skills:
            self.assertIn(expected_skill, skill_names)
        
        # Check categories are properly assigned
        categories_found = set(skill.category for skill in result.skills)
        self.assertIn(SkillCategory.TECHNICAL, categories_found)
        self.assertIn(SkillCategory.FRAMEWORK, categories_found)
        self.assertIn(SkillCategory.DATABASE, categories_found)
        
        # Check confidence scores are reasonable
        for skill in result.skills:
            self.assertGreaterEqual(skill.confidence, 0.0)
            self.assertLessEqual(skill.confidence, 1.0)
    
    def test_get_skill_domains(self):
        """Test skill domain grouping."""
        # Create mock extraction result
        skills = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 0),
            SkillMatch("JavaScript", SkillCategory.TECHNICAL, 0.8, "context", 10),
            SkillMatch("React", SkillCategory.FRAMEWORK, 0.85, "context", 20),
            SkillMatch("MySQL", SkillCategory.DATABASE, 0.8, "context", 30),
            SkillMatch("AWS", SkillCategory.CLOUD, 0.9, "context", 40)
        ]
        
        result = ExtractionResult(
            skills=skills,
            total_skills=5,
            categories={},
            confidence_distribution={},
            processing_time=0.1
        )
        
        domains = self.analyzer.get_skill_domains(result)
        
        self.assertIn("Programming Languages", domains)
        self.assertIn("Web Frameworks", domains)
        self.assertIn("Databases", domains)
        self.assertIn("Cloud Platforms", domains)
        
        self.assertIn("Python", domains["Programming Languages"])
        self.assertIn("React", domains["Web Frameworks"])
        self.assertIn("MySQL", domains["Databases"])
        self.assertIn("AWS", domains["Cloud Platforms"])
    
    def test_enhance_skill_matches(self):
        """Test skill match enhancement."""
        original_matches = [
            SkillMatch(
                "Python", 
                SkillCategory.TECHNICAL, 
                0.7, 
                "5 years of expert experience with Python programming", 
                0
            )
        ]
        
        text = "Full resume text with Python and related technologies"
        enhanced = self.analyzer._enhance_skill_matches(original_matches, text)
        
        # Should have enhanced confidence due to years and proficiency
        self.assertGreater(enhanced[0].confidence, 0.7)
    
    def test_find_related_skills(self):
        """Test finding related skills near a match."""
        skill_match = SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 50)
        text = "I use Python with Django framework and MySQL database for web development"
        
        related = self.analyzer._find_related_skills(skill_match, text)
        
        # Should find Django and MySQL as related skills
        self.assertTrue(len(related) > 0)
        # Note: This test might need adjustment based on the exact implementation


class TestTechnicalAnalyzerIntegration(unittest.TestCase):
    """Integration tests for TechnicalSkillsAnalyzer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Use a smaller database for integration tests
        self.small_database = {
            "Python": {"category": "programming_languages", "domain": "Programming Languages"},
            "React": {"category": "frameworks", "domain": "Web Frameworks"},
            "MySQL": {"category": "databases", "domain": "Databases"}
        }
        self.analyzer = TechnicalSkillsAnalyzer(self.small_database)
    
    def test_full_extraction_workflow(self):
        """Test the complete extraction workflow."""
        resume_text = """
        TECHNICAL SKILLS
        - Python (3 years experience)
        - React for frontend development
        - MySQL database management
        """
        
        result = self.analyzer.extract_skills(resume_text)
        
        # Verify result structure
        self.assertIsInstance(result, ExtractionResult)
        self.assertGreater(result.total_skills, 0)
        self.assertIn('extractor_type', result.metadata)
        self.assertEqual(result.metadata['extractor_type'], 'TechnicalSkillsAnalyzer')
        
        # Verify skills were found
        skill_names = [skill.skill for skill in result.skills]
        self.assertIn("Python", skill_names)
        
        # Verify statistics
        stats = self.analyzer.get_skill_statistics(result)
        self.assertIn('total_skills', stats)
        self.assertIn('category_breakdown', stats)


if __name__ == '__main__':
    unittest.main()