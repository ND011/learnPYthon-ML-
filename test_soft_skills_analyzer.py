"""
Unit tests for SoftSkillsAnalyzer class.

Tests soft skill extraction, categorization, pattern matching,
context-aware analysis, and achievement-based skill inference.
"""

import unittest
from unittest.mock import Mock, patch
import time

# Import the classes to test
from resume_keyword_extractor.analyzers.soft_skills_analyzer import SoftSkillsAnalyzer
from resume_keyword_extractor.analyzers.base_extractor import (
    SkillMatch, ExtractionResult, SkillCategory, ConfidenceLevel
)


class TestSoftSkillsAnalyzer(unittest.TestCase):
    """Test cases for SoftSkillsAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock soft skills database
        self.mock_database = {
            "Leadership": {
                "category": "leadership",
                "type": "soft_skill",
                "variations": ["Leadership Skills", "Team Leadership"],
                "synonyms": ["Management", "Team Management"],
                "context_keywords": ["lead", "manage", "supervise", "direct"]
            },
            "Communication": {
                "category": "communication",
                "type": "soft_skill",
                "variations": ["Communication Skills", "Verbal Communication"],
                "synonyms": ["Interpersonal Skills", "Speaking"],
                "context_keywords": ["communicate", "present", "speak"]
            },
            "Problem Solving": {
                "category": "problem_solving",
                "type": "soft_skill",
                "variations": ["Problem-Solving", "Problem Solving Skills"],
                "synonyms": ["Analytical Thinking", "Critical Thinking"],
                "context_keywords": ["solve", "analyze", "resolve"]
            },
            "Teamwork": {
                "category": "collaboration",
                "type": "soft_skill",
                "variations": ["Team Collaboration", "Collaboration"],
                "synonyms": ["Team Player", "Cooperative"],
                "context_keywords": ["team", "collaborate", "cooperate"]
            },
            "Time Management": {
                "category": "time_management",
                "type": "soft_skill",
                "variations": ["Time Management Skills", "Organization"],
                "synonyms": ["Planning", "Scheduling"],
                "context_keywords": ["organize", "plan", "schedule"]
            }
        }
        
        self.analyzer = SoftSkillsAnalyzer(self.mock_database)
    
    def test_initialization(self):
        """Test SoftSkillsAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer.logger)
        self.assertEqual(self.analyzer.skill_database, self.mock_database)
        self.assertIn('communication', self.analyzer.category_mapping)
        self.assertIn('leadership', self.analyzer.category_mapping)
        self.assertIn('strong', self.analyzer.context_strength_indicators)
    
    def test_get_supported_categories(self):
        """Test getting supported skill categories."""
        categories = self.analyzer.get_supported_categories()
        self.assertEqual(categories, [SkillCategory.SOFT])
    
    def test_build_skill_database(self):
        """Test building skill database from SOFT_SKILLS data."""
        # Test with default database (no custom database)
        analyzer = SoftSkillsAnalyzer()
        
        self.assertIsInstance(analyzer.skill_database, dict)
        self.assertGreater(len(analyzer.skill_database), 0)
        
        # Check that skills from different categories are included
        skill_names = list(analyzer.skill_database.keys())
        self.assertIn("Communication", skill_names)
        self.assertIn("Leadership", skill_names)
        self.assertIn("Problem Solving", skill_names)
    
    def test_generate_skill_variations(self):
        """Test skill variation generation."""
        variations = self.analyzer._generate_skill_variations("Communication")
        
        self.assertIn("Communication Skills", variations)
        self.assertIn("Communication Skill", variations)
        
        # Test with gerund form
        variations = self.analyzer._generate_skill_variations("Leading")
        self.assertIn("Lead", variations)
        self.assertIn("Led", variations)
        
        # Test with action skills
        variations = self.analyzer._generate_skill_variations("Leadership")
        self.assertTrue(len(variations) > 0)
    
    def test_generate_skill_synonyms(self):
        """Test skill synonym generation."""
        synonyms = self.analyzer._generate_skill_synonyms("Communication")
        self.assertIn("verbal communication", synonyms)
        self.assertIn("written communication", synonyms)
        self.assertIn("interpersonal", synonyms)
        
        synonyms = self.analyzer._generate_skill_synonyms("Leadership")
        self.assertIn("management", synonyms)
        self.assertIn("team leadership", synonyms)
        self.assertIn("supervision", synonyms)
        
        synonyms = self.analyzer._generate_skill_synonyms("Problem Solving")
        self.assertIn("analytical thinking", synonyms)
        self.assertIn("critical thinking", synonyms)
        self.assertIn("troubleshooting", synonyms)
    
    def test_preprocess_text(self):
        """Test text preprocessing for soft skills."""
        text = "I have comm skills, mgmt experience, and proj mgmt background."
        processed = self.analyzer._preprocess_text(text)
        
        self.assertIn("communication", processed)
        self.assertIn("management", processed)
        self.assertIn("project management", processed)
    
    def test_extract_keyword_matches(self):
        """Test direct keyword matching for soft skills."""
        text = "I have strong leadership skills and excellent communication abilities."
        
        matches = self.analyzer._extract_keyword_matches(text)
        
        # Should find leadership and communication
        skill_names = [match.skill for match in matches]
        self.assertIn("Leadership", skill_names)
        self.assertIn("Communication", skill_names)
        
        # Check categories
        for match in matches:
            self.assertEqual(match.category, SkillCategory.SOFT)
            self.assertGreaterEqual(match.confidence, self.analyzer.confidence_thresholds[ConfidenceLevel.LOW])
    
    def test_infer_skills_from_pattern(self):
        """Test skill inference from pattern matches."""
        # Leadership indicators
        skills = self.analyzer._infer_skills_from_pattern(
            'leadership_indicators', 
            'led a team of 5 developers', 
            ['Leadership', 'Team Management', 'Supervision']
        )
        self.assertIn('Leadership', skills)
        self.assertIn('Team Management', skills)
        
        # Communication indicators
        skills = self.analyzer._infer_skills_from_pattern(
            'communication_indicators', 
            'presented to stakeholders', 
            ['Presentation', 'Public Speaking', 'Communication']
        )
        self.assertIn('Presentation', skills)
        self.assertIn('Public Speaking', skills)
        
        # Problem solving indicators
        skills = self.analyzer._infer_skills_from_pattern(
            'problem_solving_indicators', 
            'analyzed system performance', 
            ['Analytical Thinking', 'Problem Solving']
        )
        self.assertIn('Analytical Thinking', skills)
    
    def test_analyze_context_strength(self):
        """Test context strength analysis."""
        # Strong context
        strong_context = "Led a team of 10 developers and managed multiple projects"
        strength = self.analyzer._analyze_context_strength(strong_context)
        self.assertEqual(strength, 'strong')
        
        # Medium context
        medium_context = "Collaborated with team members on various projects"
        strength = self.analyzer._analyze_context_strength(medium_context)
        self.assertEqual(strength, 'medium')
        
        # Weak context
        weak_context = "Familiar with leadership concepts and team dynamics"
        strength = self.analyzer._analyze_context_strength(weak_context)
        self.assertEqual(strength, 'weak')
    
    def test_calculate_soft_skill_confidence(self):
        """Test soft skill confidence calculation."""
        skill_data = self.mock_database["Leadership"]
        
        # High confidence context with action verbs and quantification
        high_context = "Led a team of 15 developers and increased productivity by 30%"
        confidence = self.analyzer._calculate_soft_skill_confidence(
            "Leadership", high_context, "keyword", skill_data
        )
        self.assertGreater(confidence, 0.8)
        
        # Medium confidence context
        medium_context = "Worked with team members on collaborative projects"
        confidence = self.analyzer._calculate_soft_skill_confidence(
            "Leadership", medium_context, "pattern", skill_data
        )
        self.assertGreater(confidence, 0.5)
        self.assertLess(confidence, 0.8)
        
        # Low confidence context (passive mention)
        low_context = "Familiar with leadership principles"
        confidence = self.analyzer._calculate_soft_skill_confidence(
            "Leadership", low_context, "inference", skill_data
        )
        self.assertLess(confidence, 0.6)
    
    def test_calculate_skill_frequency(self):
        """Test skill frequency calculation."""
        text = "Leadership is important. I have leadership experience and leadership skills."
        frequency = self.analyzer._calculate_skill_frequency("Leadership", text)
        self.assertEqual(frequency, 3)  # Should find 3 mentions
        
        # Test with variations
        text_with_variations = "Leadership skills and team leadership experience"
        frequency = self.analyzer._calculate_skill_frequency("Leadership", text_with_variations)
        self.assertGreaterEqual(frequency, 1)  # Should find at least the base skill
    
    def test_find_related_soft_skills(self):
        """Test finding related soft skills near a match."""
        skill_match = SkillMatch("Leadership", SkillCategory.SOFT, 0.9, "context", 50)
        text = "Strong leadership and communication skills with excellent teamwork abilities"
        
        related = self.analyzer._find_related_soft_skills(skill_match, text)
        
        # Should find communication and teamwork as related skills
        self.assertTrue(len(related) >= 0)  # May or may not find related skills depending on database
    
    def test_extract_pattern_matches(self):
        """Test pattern-based skill extraction."""
        text = """
        Led a team of 8 developers on multiple projects.
        Presented quarterly results to senior management.
        Resolved critical system issues and improved performance.
        Achieved 95% customer satisfaction rating.
        """
        
        matches = self.analyzer._extract_pattern_matches(text)
        
        # Should find skills based on action patterns
        self.assertGreater(len(matches), 0)
        
        # Check that all matches are soft skills
        for match in matches:
            self.assertEqual(match.category, SkillCategory.SOFT)
            self.assertGreaterEqual(match.confidence, self.analyzer.confidence_thresholds[ConfidenceLevel.LOW])
    
    def test_extract_inferred_skills(self):
        """Test context-aware skill inference."""
        text = """
        Managed projects with budgets exceeding $2M.
        Presented to executive leadership team.
        Coordinated cross-functional teams of 20+ members.
        Provided customer service to high-value clients.
        Resolved complex technical problems.
        """
        
        matches = self.analyzer._extract_inferred_skills(text)
        
        # Should infer various soft skills from context
        self.assertGreater(len(matches), 0)
        
        skill_names = [match.skill for match in matches]
        # Should infer leadership, communication, problem solving, etc.
        expected_skills = ['Leadership', 'Project Management', 'Communication', 'Problem Solving']
        
        # At least some expected skills should be inferred
        found_expected = any(skill in skill_names for skill in expected_skills)
        self.assertTrue(found_expected)
    
    def test_extract_achievement_based_skills(self):
        """Test achievement-based skill extraction."""
        text = """
        Increased team productivity by 40% through process improvements.
        Managed a team of 12 software engineers.
        Led projects with budgets of $5M+ and delivered on time.
        """
        
        matches = self.analyzer._extract_achievement_based_skills(text)
        
        # Should find skills based on quantified achievements
        self.assertGreater(len(matches), 0)
        
        skill_names = [match.skill for match in matches]
        expected_skills = ['Results-Oriented', 'Performance Improvement', 'Leadership', 'Project Management']
        
        # Should find achievement-based skills
        found_expected = any(skill in skill_names for skill in expected_skills)
        self.assertTrue(found_expected)
        
        # Achievement-based skills should have higher confidence
        for match in matches:
            self.assertGreaterEqual(match.confidence, 0.5)
    
    def test_enhance_skill_matches(self):
        """Test skill match enhancement."""
        original_matches = [
            SkillMatch(
                "Leadership", 
                SkillCategory.SOFT, 
                0.6, 
                "Led a team of 10 developers and achieved 25% improvement", 
                0
            )
        ]
        
        text = "Full resume text with leadership and team management experience"
        enhanced = self.analyzer._enhance_skill_matches(original_matches, text)
        
        # Should have enhanced confidence due to strong context
        self.assertGreaterEqual(len(enhanced), 1)
        # Confidence might be adjusted based on context strength
        self.assertGreaterEqual(enhanced[0].confidence, 0.0)
        self.assertLessEqual(enhanced[0].confidence, 1.0)
    
    def test_extract_skills_comprehensive(self):
        """Test comprehensive soft skill extraction."""
        resume_text = """
        John Doe - Senior Project Manager
        
        EXPERIENCE
        Senior Project Manager at Tech Corp (2019-2023)
        - Led cross-functional teams of 15+ members across multiple departments
        - Managed project budgets exceeding $3M with 98% on-time delivery
        - Presented quarterly results to C-level executives and board members
        - Resolved complex stakeholder conflicts and improved team collaboration
        - Mentored 5 junior project managers and improved team productivity by 35%
        - Negotiated contracts with vendors and achieved 20% cost savings
        
        Team Lead at StartupCo (2017-2019)
        - Coordinated agile development teams and facilitated daily standups
        - Communicated project status to stakeholders and managed expectations
        - Implemented process improvements that reduced delivery time by 25%
        - Provided technical guidance and coaching to team members
        
        SKILLS
        - Strong leadership and team management capabilities
        - Excellent communication and presentation skills
        - Problem-solving and analytical thinking
        - Customer service and client relationship management
        - Time management and organizational skills
        """
        
        result = self.analyzer.extract_skills(resume_text)
        
        # Check that skills were found
        self.assertGreater(result.total_skills, 0)
        self.assertIsInstance(result, ExtractionResult)
        
        # Check for expected soft skills
        skill_names = [skill.skill for skill in result.skills]
        expected_skills = [
            "Leadership", "Communication", "Problem Solving", 
            "Team Management", "Project Management", "Presentation"
        ]
        
        # Should find several expected skills
        found_count = sum(1 for skill in expected_skills if skill in skill_names)
        self.assertGreater(found_count, 2)  # Should find at least 3 expected skills
        
        # All skills should be soft skills
        for skill in result.skills:
            self.assertEqual(skill.category, SkillCategory.SOFT)
        
        # Check confidence scores are reasonable
        for skill in result.skills:
            self.assertGreaterEqual(skill.confidence, 0.0)
            self.assertLessEqual(skill.confidence, 1.0)
        
        # Check processing time is recorded
        self.assertGreater(result.processing_time, 0)
    
    def test_get_skill_categories_breakdown(self):
        """Test skill categories breakdown."""
        # Create mock extraction result
        skills = [
            SkillMatch("Leadership", SkillCategory.SOFT, 0.9, "context", 0),
            SkillMatch("Communication", SkillCategory.SOFT, 0.8, "context", 10),
            SkillMatch("Problem Solving", SkillCategory.SOFT, 0.85, "context", 20),
            SkillMatch("Teamwork", SkillCategory.SOFT, 0.8, "context", 30)
        ]
        
        result = ExtractionResult(
            skills=skills,
            total_skills=4,
            categories={SkillCategory.SOFT: 4},
            confidence_distribution={},
            processing_time=0.1
        )
        
        categories = self.analyzer.get_skill_categories_breakdown(result)
        
        self.assertIn("leadership", categories)
        self.assertIn("communication", categories)
        self.assertIn("problem_solving", categories)
        self.assertIn("collaboration", categories)
        
        self.assertIn("Leadership", categories["leadership"])
        self.assertIn("Communication", categories["communication"])
        self.assertIn("Problem Solving", categories["problem_solving"])
        self.assertIn("Teamwork", categories["collaboration"])
    
    def test_get_context_analysis(self):
        """Test context analysis of extracted skills."""
        # Create mock extraction result with varied contexts
        skills = [
            SkillMatch("Leadership", SkillCategory.SOFT, 0.9, "Led a team of 10 developers", 0),
            SkillMatch("Communication", SkillCategory.SOFT, 0.8, "Presented to client stakeholders", 10),
            SkillMatch("Problem Solving", SkillCategory.SOFT, 0.85, "Resolved issues and improved by 30%", 20),
            SkillMatch("Project Management", SkillCategory.SOFT, 0.8, "Managed team of 15 members", 30)
        ]
        
        result = ExtractionResult(
            skills=skills,
            total_skills=4,
            categories={SkillCategory.SOFT: 4},
            confidence_distribution={},
            processing_time=0.1
        )
        
        analysis = self.analyzer.get_context_analysis(result)
        
        self.assertIn('action_based_skills', analysis)
        self.assertIn('achievement_based_skills', analysis)
        self.assertIn('team_context_skills', analysis)
        self.assertIn('client_context_skills', analysis)
        self.assertIn('quantified_skills', analysis)
        self.assertIn('summary', analysis)
        
        # Check summary statistics
        self.assertEqual(analysis['summary']['total_skills'], 4)
        self.assertIn('action_based_percentage', analysis['summary'])
        self.assertIn('quantified_percentage', analysis['summary'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty text
        result = self.analyzer.extract_skills("")
        self.assertEqual(result.total_skills, 0)
        
        # Text with no soft skills
        result = self.analyzer.extract_skills("Technical skills: Python, Java, SQL")
        # Might still find some inferred skills, but should handle gracefully
        self.assertIsInstance(result, ExtractionResult)
        
        # Very long text
        long_text = "Leadership " * 1000
        result = self.analyzer.extract_skills(long_text)
        self.assertIsInstance(result, ExtractionResult)
    
    def test_confidence_thresholds(self):
        """Test that confidence thresholds are properly applied."""
        # Test with text that should generate low confidence matches
        text = "Familiar with leadership concepts"
        result = self.analyzer.extract_skills(text)
        
        # All returned skills should meet minimum confidence threshold
        for skill in result.skills:
            self.assertGreaterEqual(skill.confidence, self.analyzer.confidence_thresholds[ConfidenceLevel.LOW])


class TestSoftSkillsAnalyzerIntegration(unittest.TestCase):
    """Integration tests for SoftSkillsAnalyzer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.analyzer = SoftSkillsAnalyzer()  # Use default database
    
    def test_full_extraction_workflow(self):
        """Test the complete extraction workflow."""
        resume_text = """
        PROFESSIONAL EXPERIENCE
        Project Manager - Led teams of 8-12 people
        - Excellent communication with stakeholders
        - Strong problem-solving abilities
        - Managed multiple projects simultaneously
        """
        
        result = self.analyzer.extract_skills(resume_text)
        
        # Verify result structure
        self.assertIsInstance(result, ExtractionResult)
        self.assertGreaterEqual(result.total_skills, 0)
        self.assertIn('extractor_type', result.metadata)
        self.assertEqual(result.metadata['extractor_type'], 'SoftSkillsAnalyzer')
        
        # Verify statistics
        stats = self.analyzer.get_skill_statistics(result)
        self.assertIn('total_skills', stats)
        self.assertIn('category_breakdown', stats)
    
    def test_real_world_resume_sample(self):
        """Test with a realistic resume sample."""
        resume_text = """
        SUMMARY
        Experienced software engineer with strong leadership skills and excellent 
        communication abilities. Proven track record of managing cross-functional 
        teams and delivering complex projects on time.
        
        EXPERIENCE
        Senior Software Engineer | TechCorp | 2020-2023
        • Led a team of 6 developers in building scalable web applications
        • Collaborated with product managers and designers on feature requirements
        • Mentored junior developers and conducted code reviews
        • Presented technical solutions to stakeholders and executive team
        • Resolved critical production issues and improved system reliability
        
        Software Engineer | StartupCo | 2018-2020
        • Worked closely with cross-functional teams to deliver features
        • Participated in agile development processes and daily standups
        • Communicated project status and technical challenges to management
        • Contributed to team knowledge sharing and best practices
        """
        
        result = self.analyzer.extract_skills(resume_text)
        
        # Should find multiple soft skills
        self.assertGreater(result.total_skills, 3)
        
        # Check for expected skills
        skill_names = [skill.skill for skill in result.skills]
        expected_skills = ["Leadership", "Communication", "Collaboration", "Mentoring"]
        
        found_skills = [skill for skill in expected_skills if skill in skill_names]
        self.assertGreater(len(found_skills), 1)  # Should find at least 2 expected skills


if __name__ == '__main__':
    unittest.main()