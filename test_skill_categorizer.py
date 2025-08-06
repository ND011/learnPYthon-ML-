"""
Unit tests for SkillCategorizer class.

Tests skill grouping, hierarchy management, deduplication,
normalization, frequency analysis, and importance scoring.
"""

import unittest
from unittest.mock import Mock, patch
import time

# Import the classes to test
from resume_keyword_extractor.analyzers.skill_categorizer import (
    SkillCategorizer, SkillGroup, SkillHierarchy, CategorizationResult, SkillImportance
)
from resume_keyword_extractor.analyzers.base_extractor import (
    SkillMatch, ExtractionResult, SkillCategory, ConfidenceLevel
)


class TestSkillCategorizer(unittest.TestCase):
    """Test cases for SkillCategorizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.categorizer = SkillCategorizer()
        
        # Create sample skills for testing
        self.sample_skills = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "Expert in Python programming", 0),
            SkillMatch("JavaScript", SkillCategory.TECHNICAL, 0.85, "Proficient in JavaScript", 10),
            SkillMatch("React", SkillCategory.TECHNICAL, 0.8, "Experience with React framework", 20),
            SkillMatch("Leadership", SkillCategory.SOFT, 0.9, "Led team of 10 developers", 30),
            SkillMatch("Communication", SkillCategory.SOFT, 0.85, "Excellent communication skills", 40),
            SkillMatch("Git", SkillCategory.TOOL, 0.8, "Version control with Git", 50),
            SkillMatch("AWS", SkillCategory.CLOUD, 0.85, "Cloud deployment on AWS", 60),
            SkillMatch("MySQL", SkillCategory.DATABASE, 0.8, "Database management with MySQL", 70)
        ]
        
        # Create sample extraction results
        self.sample_extraction_results = [
            ExtractionResult(
                skills=self.sample_skills[:4],
                total_skills=4,
                categories={SkillCategory.TECHNICAL: 3, SkillCategory.SOFT: 1},
                confidence_distribution={},
                processing_time=0.1
            ),
            ExtractionResult(
                skills=self.sample_skills[4:],
                total_skills=4,
                categories={SkillCategory.SOFT: 1, SkillCategory.TOOL: 1, 
                          SkillCategory.CLOUD: 1, SkillCategory.DATABASE: 1},
                confidence_distribution={},
                processing_time=0.1
            )
        ]
    
    def test_initialization(self):
        """Test SkillCategorizer initialization."""
        self.assertIsNotNone(self.categorizer.logger)
        self.assertIsInstance(self.categorizer.default_hierarchies, dict)
        self.assertIsInstance(self.categorizer.normalization_rules, dict)
        self.assertIn(SkillCategory.TECHNICAL, self.categorizer.default_hierarchies)
        self.assertIn(SkillCategory.SOFT, self.categorizer.default_hierarchies)
    
    def test_build_default_hierarchies(self):
        """Test building default skill hierarchies."""
        hierarchies = self.categorizer._build_default_hierarchies()
        
        self.assertIn(SkillCategory.TECHNICAL, hierarchies)
        self.assertIn(SkillCategory.SOFT, hierarchies)
        self.assertIn(SkillCategory.TOOL, hierarchies)
        self.assertIn(SkillCategory.CLOUD, hierarchies)
        
        # Check technical skills structure
        tech_hierarchy = hierarchies[SkillCategory.TECHNICAL]
        self.assertIn('programming_languages', tech_hierarchy)
        self.assertIn('web_technologies', tech_hierarchy)
        self.assertIn('Python', tech_hierarchy['programming_languages'])
        self.assertIn('JavaScript', tech_hierarchy['programming_languages'])
        
        # Check soft skills structure
        soft_hierarchy = hierarchies[SkillCategory.SOFT]
        self.assertIn('leadership', soft_hierarchy)
        self.assertIn('communication', soft_hierarchy)
        self.assertIn('Leadership', soft_hierarchy['leadership'])
        self.assertIn('Communication', soft_hierarchy['communication'])
    
    def test_build_normalization_rules(self):
        """Test building skill normalization rules."""
        rules = self.categorizer._build_normalization_rules()
        
        self.assertIsInstance(rules, dict)
        self.assertEqual(rules['javascript'], 'JavaScript')
        self.assertEqual(rules['js'], 'JavaScript')
        self.assertEqual(rules['python3'], 'Python')
        self.assertEqual(rules['c++'], 'C++')
        self.assertEqual(rules['reactjs'], 'React')
        self.assertEqual(rules['nodejs'], 'Node.js')
        self.assertEqual(rules['mysql'], 'MySQL')
        self.assertEqual(rules['aws'], 'AWS')
    
    def test_normalize_skill_name(self):
        """Test skill name normalization."""
        # Test normalization rules
        self.assertEqual(self.categorizer._normalize_skill_name('javascript'), 'JavaScript')
        self.assertEqual(self.categorizer._normalize_skill_name('JS'), 'JavaScript')
        self.assertEqual(self.categorizer._normalize_skill_name('python3'), 'Python')
        self.assertEqual(self.categorizer._normalize_skill_name('reactjs'), 'React')
        self.assertEqual(self.categorizer._normalize_skill_name('nodejs'), 'Node.js')
        
        # Test general formatting
        self.assertEqual(self.categorizer._normalize_skill_name('machine learning'), 'Machine Learning')
        self.assertEqual(self.categorizer._normalize_skill_name('data science'), 'Data Science')
        
        # Test special cases
        self.assertEqual(self.categorizer._normalize_skill_name('html5'), 'HTML5')
        self.assertEqual(self.categorizer._normalize_skill_name('css3'), 'CSS3')
        self.assertEqual(self.categorizer._normalize_skill_name('rest api'), 'Rest API')
        self.assertEqual(self.categorizer._normalize_skill_name('json parsing'), 'JSON Parsing')
    
    def test_normalize_and_deduplicate(self):
        """Test skill normalization and deduplication."""
        # Create skills with duplicates and variations
        skills_with_duplicates = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 0),
            SkillMatch("python", SkillCategory.TECHNICAL, 0.8, "context", 10),
            SkillMatch("Python3", SkillCategory.TECHNICAL, 0.85, "context", 20),
            SkillMatch("JavaScript", SkillCategory.TECHNICAL, 0.9, "context", 30),
            SkillMatch("js", SkillCategory.TECHNICAL, 0.7, "context", 40),
            SkillMatch("React", SkillCategory.TECHNICAL, 0.8, "context", 50),
            SkillMatch("reactjs", SkillCategory.TECHNICAL, 0.75, "context", 60)
        ]
        
        normalized_mapping, deduplicated, duplicates_count = self.categorizer._normalize_and_deduplicate(skills_with_duplicates)
        
        # Check that duplicates were removed
        self.assertEqual(duplicates_count, 4)  # python, Python3, js, reactjs are duplicates
        self.assertEqual(len(deduplicated), 3)  # Python, JavaScript, React
        
        # Check that highest confidence versions were kept
        skill_names = [skill.skill for skill in deduplicated]
        self.assertIn("Python", skill_names)
        self.assertIn("JavaScript", skill_names)
        self.assertIn("React", skill_names)
        
        # Check that normalization mapping is correct
        self.assertEqual(normalized_mapping["python"], "Python")
        self.assertEqual(normalized_mapping["Python3"], "Python")
        self.assertEqual(normalized_mapping["js"], "JavaScript")
        self.assertEqual(normalized_mapping["reactjs"], "React")
        
        # Check that highest confidence skill was kept for Python
        python_skill = next(skill for skill in deduplicated if skill.skill == "Python")
        self.assertEqual(python_skill.confidence, 0.9)  # Original "Python" had highest confidence
    
    def test_group_skills(self):
        """Test skill grouping functionality."""
        groups = self.categorizer._group_skills(self.sample_skills)
        
        self.assertGreater(len(groups), 0)
        
        # Check that groups have correct structure
        for group in groups:
            self.assertIsInstance(group, SkillGroup)
            self.assertIsInstance(group.name, str)
            self.assertIsInstance(group.category, SkillCategory)
            self.assertIsInstance(group.skills, list)
            self.assertGreater(len(group.skills), 0)
        
        # Check that skills are grouped by category
        categories_found = set(group.category for group in groups)
        expected_categories = set(skill.category for skill in self.sample_skills)
        self.assertEqual(categories_found, expected_categories)
    
    def test_group_by_subcategory(self):
        """Test grouping skills by subcategory."""
        # Test with technical skills
        tech_skills = [skill for skill in self.sample_skills if skill.category == SkillCategory.TECHNICAL]
        subcategory_groups = self.categorizer._group_by_subcategory(SkillCategory.TECHNICAL, tech_skills)
        
        self.assertIsInstance(subcategory_groups, dict)
        self.assertGreater(len(subcategory_groups), 0)
        
        # Check that Python and JavaScript are in programming_languages
        if 'programming_languages' in subcategory_groups:
            prog_lang_skills = [skill.skill for skill in subcategory_groups['programming_languages']]
            self.assertIn('Python', prog_lang_skills)
            self.assertIn('JavaScript', prog_lang_skills)
        
        # Test with soft skills
        soft_skills = [skill for skill in self.sample_skills if skill.category == SkillCategory.SOFT]
        if soft_skills:
            subcategory_groups = self.categorizer._group_by_subcategory(SkillCategory.SOFT, soft_skills)
            self.assertIsInstance(subcategory_groups, dict)
    
    def test_build_hierarchies(self):
        """Test building skill hierarchies."""
        groups = self.categorizer._group_skills(self.sample_skills)
        hierarchies = self.categorizer._build_hierarchies(groups)
        
        self.assertIsInstance(hierarchies, dict)
        
        # Check that hierarchies are created for categories present in skills
        categories_in_skills = set(skill.category for skill in self.sample_skills)
        for category in categories_in_skills:
            self.assertIn(category, hierarchies)
            
            hierarchy = hierarchies[category]
            self.assertIsInstance(hierarchy, SkillHierarchy)
            self.assertEqual(hierarchy.root_category, category)
            self.assertIsInstance(hierarchy.subcategories, dict)
            self.assertIsInstance(hierarchy.skill_relationships, dict)
    
    def test_calculate_importance_scores(self):
        """Test importance score calculation."""
        importance_scores = self.categorizer._calculate_importance_scores(self.sample_skills)
        
        self.assertIsInstance(importance_scores, dict)
        self.assertEqual(len(importance_scores), len(self.sample_skills))
        
        # Check that all skills have scores
        for skill in self.sample_skills:
            self.assertIn(skill.skill, importance_scores)
            score = importance_scores[skill.skill]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Skills with higher confidence should generally have higher importance
        python_score = importance_scores.get("Python", 0)
        leadership_score = importance_scores.get("Leadership", 0)
        self.assertGreater(python_score, 0.5)  # High confidence skill
        self.assertGreater(leadership_score, 0.5)  # High confidence skill
    
    def test_analyze_context_strength(self):
        """Test context strength analysis."""
        # Strong context
        strong_context = "Expert in Python with 5 years of experience, led multiple projects"
        strength = self.categorizer._analyze_context_strength(strong_context)
        self.assertEqual(strength, 0.9)
        
        # Medium context
        medium_context = "Worked with JavaScript on various projects"
        strength = self.categorizer._analyze_context_strength(medium_context)
        self.assertEqual(strength, 0.6)
        
        # Weak context
        weak_context = "Basic knowledge of HTML and CSS"
        strength = self.categorizer._analyze_context_strength(weak_context)
        self.assertEqual(strength, 0.3)
        
        # Empty context
        strength = self.categorizer._analyze_context_strength("")
        self.assertEqual(strength, 0.5)
    
    def test_analyze_frequency(self):
        """Test frequency analysis."""
        # Create skills with duplicates for frequency testing
        skills_with_frequency = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 0),
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.8, "context", 10),
            SkillMatch("JavaScript", SkillCategory.TECHNICAL, 0.9, "context", 20),
            SkillMatch("React", SkillCategory.TECHNICAL, 0.8, "context", 30)
        ]
        
        frequency_analysis = self.categorizer._analyze_frequency(skills_with_frequency)
        
        self.assertIsInstance(frequency_analysis, dict)
        self.assertEqual(frequency_analysis["Python"], 2)
        self.assertEqual(frequency_analysis["JavaScript"], 1)
        self.assertEqual(frequency_analysis["React"], 1)
    
    def test_enhance_skill_groups(self):
        """Test skill group enhancement."""
        groups = self.categorizer._group_skills(self.sample_skills)
        importance_scores = self.categorizer._calculate_importance_scores(self.sample_skills)
        frequency_analysis = self.categorizer._analyze_frequency(self.sample_skills)
        
        # Enhance groups
        self.categorizer._enhance_skill_groups(groups, importance_scores, frequency_analysis)
        
        # Check that groups were enhanced
        for group in groups:
            self.assertIsInstance(group.importance, SkillImportance)
            self.assertGreaterEqual(group.frequency, 1)
            self.assertIsInstance(group.related_groups, list)
            
            # Check that average confidence was calculated
            if group.skills:
                expected_avg = sum(skill.confidence for skill in group.skills) / len(group.skills)
                self.assertAlmostEqual(group.avg_confidence, expected_avg, places=1)
    
    def test_find_related_groups(self):
        """Test finding related skill groups."""
        groups = self.categorizer._group_skills(self.sample_skills)
        
        if len(groups) > 1:
            target_group = groups[0]
            related_groups = self.categorizer._find_related_groups(target_group, groups)
            
            self.assertIsInstance(related_groups, list)
            self.assertLessEqual(len(related_groups), 3)  # Limited to top 3
            
            # Related groups should not include the target group itself
            self.assertNotIn(target_group.name, related_groups)
    
    def test_categorize_skills_comprehensive(self):
        """Test comprehensive skill categorization."""
        result = self.categorizer.categorize_skills(self.sample_extraction_results)
        
        # Check result structure
        self.assertIsInstance(result, CategorizationResult)
        self.assertIsInstance(result.hierarchies, dict)
        self.assertIsInstance(result.skill_groups, list)
        self.assertIsInstance(result.normalized_skills, dict)
        self.assertIsInstance(result.importance_scores, dict)
        self.assertIsInstance(result.frequency_analysis, dict)
        self.assertGreater(result.processing_time, 0)
        
        # Check that hierarchies were created
        self.assertGreater(len(result.hierarchies), 0)
        
        # Check that skill groups were created
        self.assertGreater(len(result.skill_groups), 0)
        
        # Check that skills were processed
        self.assertGreater(len(result.normalized_skills), 0)
        self.assertGreater(len(result.importance_scores), 0)
        self.assertGreater(len(result.frequency_analysis), 0)
        
        # Check metadata
        self.assertIn('total_original_skills', result.metadata)
        self.assertIn('total_deduplicated_skills', result.metadata)
        self.assertIn('total_groups', result.metadata)
        self.assertIn('hierarchies_count', result.metadata)
    
    def test_get_categorization_summary(self):
        """Test categorization summary generation."""
        result = self.categorizer.categorize_skills(self.sample_extraction_results)
        summary = self.categorizer.get_categorization_summary(result)
        
        # Check summary structure
        self.assertIn('total_skills', summary)
        self.assertIn('total_groups', summary)
        self.assertIn('duplicates_removed', summary)
        self.assertIn('processing_time', summary)
        self.assertIn('categories', summary)
        self.assertIn('importance_distribution', summary)
        self.assertIn('top_skills_by_importance', summary)
        self.assertIn('most_frequent_skills', summary)
        
        # Check categories breakdown
        self.assertIsInstance(summary['categories'], dict)
        for category_name, category_info in summary['categories'].items():
            self.assertIn('skill_count', category_info)
            self.assertIn('subcategories', category_info)
            self.assertIn('avg_confidence', category_info)
            self.assertIn('top_skills', category_info)
        
        # Check importance distribution
        self.assertIsInstance(summary['importance_distribution'], dict)
        
        # Check top skills lists
        self.assertIsInstance(summary['top_skills_by_importance'], list)
        self.assertIsInstance(summary['most_frequent_skills'], list)
        
        # Check that top skills have correct structure
        if summary['top_skills_by_importance']:
            top_skill = summary['top_skills_by_importance'][0]
            self.assertIn('skill', top_skill)
            self.assertIn('score', top_skill)
        
        if summary['most_frequent_skills']:
            frequent_skill = summary['most_frequent_skills'][0]
            self.assertIn('skill', frequent_skill)
            self.assertIn('frequency', frequent_skill)
    
    def test_skill_group_dataclass(self):
        """Test SkillGroup dataclass functionality."""
        skills = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 0),
            SkillMatch("Django", SkillCategory.TECHNICAL, 0.8, "context", 10)
        ]
        
        group = SkillGroup(
            name="python_ecosystem",
            category=SkillCategory.TECHNICAL,
            skills=skills
        )
        
        # Check that post_init calculated derived properties
        self.assertEqual(group.frequency, 2)
        self.assertEqual(group.avg_confidence, 0.85)  # (0.9 + 0.8) / 2
    
    def test_skill_hierarchy_dataclass(self):
        """Test SkillHierarchy dataclass functionality."""
        hierarchy = SkillHierarchy(root_category=SkillCategory.TECHNICAL)
        
        # Test adding groups
        group1 = SkillGroup("programming_languages", SkillCategory.TECHNICAL)
        group2 = SkillGroup("web_frameworks", SkillCategory.TECHNICAL)
        
        hierarchy.add_group("programming", group1)
        hierarchy.add_group("web", group2)
        
        self.assertIn("programming", hierarchy.subcategories)
        self.assertIn("web", hierarchy.subcategories)
        self.assertEqual(len(hierarchy.subcategories["programming"]), 1)
        self.assertEqual(len(hierarchy.subcategories["web"]), 1)
        
        # Test get_all_skills
        skills = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 0),
            SkillMatch("React", SkillCategory.TECHNICAL, 0.8, "context", 10)
        ]
        group1.skills = [skills[0]]
        group2.skills = [skills[1]]
        
        all_skills = hierarchy.get_all_skills()
        self.assertEqual(len(all_skills), 2)
        self.assertIn(skills[0], all_skills)
        self.assertIn(skills[1], all_skills)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty extraction results
        empty_results = []
        result = self.categorizer.categorize_skills(empty_results)
        self.assertIsInstance(result, CategorizationResult)
        self.assertEqual(len(result.skill_groups), 0)
        self.assertEqual(len(result.hierarchies), 0)
        
        # Single skill
        single_skill_result = ExtractionResult(
            skills=[SkillMatch("Python", SkillCategory.TECHNICAL, 0.9, "context", 0)],
            total_skills=1,
            categories={SkillCategory.TECHNICAL: 1},
            confidence_distribution={},
            processing_time=0.1
        )
        
        result = self.categorizer.categorize_skills([single_skill_result])
        self.assertIsInstance(result, CategorizationResult)
        self.assertGreater(len(result.skill_groups), 0)
        self.assertGreater(len(result.hierarchies), 0)
    
    def test_custom_hierarchies(self):
        """Test custom hierarchies functionality."""
        custom_hierarchies = {
            SkillCategory.TECHNICAL: {
                'custom_category': ['CustomSkill1', 'CustomSkill2']
            }
        }
        
        categorizer = SkillCategorizer(custom_hierarchies=custom_hierarchies)
        self.assertEqual(categorizer.custom_hierarchies, custom_hierarchies)
    
    def test_importance_weights_configuration(self):
        """Test that importance weights are properly configured."""
        weights = self.categorizer.importance_weights
        
        # Check that all expected weights are present
        expected_weights = ['confidence', 'frequency', 'context_strength', 'category_priority', 'recency']
        for weight in expected_weights:
            self.assertIn(weight, weights)
            self.assertGreater(weights[weight], 0)
        
        # Check that weights sum to approximately 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=1)
    
    def test_category_priorities(self):
        """Test category priority configuration."""
        priorities = self.categorizer.category_priorities
        
        # Check that all skill categories have priorities
        for category in SkillCategory:
            self.assertIn(category, priorities)
            priority = priorities[category]
            self.assertGreaterEqual(priority, 0.0)
            self.assertLessEqual(priority, 1.0)
        
        # Technical skills should have high priority
        self.assertGreaterEqual(priorities[SkillCategory.TECHNICAL], 0.9)


class TestSkillCategorizerIntegration(unittest.TestCase):
    """Integration tests for SkillCategorizer."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.categorizer = SkillCategorizer()
    
    def test_full_categorization_workflow(self):
        """Test the complete categorization workflow."""
        # Create realistic extraction results
        technical_skills = [
            SkillMatch("Python", SkillCategory.TECHNICAL, 0.95, "Expert Python developer with 5 years experience", 0),
            SkillMatch("JavaScript", SkillCategory.TECHNICAL, 0.9, "Proficient in JavaScript and ES6", 10),
            SkillMatch("React", SkillCategory.TECHNICAL, 0.85, "Built multiple React applications", 20),
            SkillMatch("Node.js", SkillCategory.TECHNICAL, 0.8, "Backend development with Node.js", 30),
            SkillMatch("MySQL", SkillCategory.DATABASE, 0.8, "Database design and optimization", 40)
        ]
        
        soft_skills = [
            SkillMatch("Leadership", SkillCategory.SOFT, 0.9, "Led team of 8 developers", 50),
            SkillMatch("Communication", SkillCategory.SOFT, 0.85, "Excellent written and verbal communication", 60),
            SkillMatch("Problem Solving", SkillCategory.SOFT, 0.8, "Strong analytical and problem-solving skills", 70)
        ]
        
        extraction_results = [
            ExtractionResult(
                skills=technical_skills,
                total_skills=5,
                categories={SkillCategory.TECHNICAL: 4, SkillCategory.DATABASE: 1},
                confidence_distribution={},
                processing_time=0.1
            ),
            ExtractionResult(
                skills=soft_skills,
                total_skills=3,
                categories={SkillCategory.SOFT: 3},
                confidence_distribution={},
                processing_time=0.1
            )
        ]
        
        # Perform categorization
        result = self.categorizer.categorize_skills(extraction_results)
        
        # Verify comprehensive results
        self.assertGreater(len(result.skill_groups), 0)
        self.assertGreater(len(result.hierarchies), 0)
        self.assertEqual(len(result.normalized_skills), 8)  # All skills should be normalized
        
        # Check that technical and soft skill hierarchies were created
        self.assertIn(SkillCategory.TECHNICAL, result.hierarchies)
        self.assertIn(SkillCategory.SOFT, result.hierarchies)
        
        # Verify summary
        summary = self.categorizer.get_categorization_summary(result)
        self.assertEqual(summary['total_skills'], 8)
        self.assertIn('technical', summary['categories'])
        self.assertIn('soft', summary['categories'])
        
        # Check that Python and Leadership have high importance scores
        python_score = result.importance_scores.get('Python', 0)
        leadership_score = result.importance_scores.get('Leadership', 0)
        self.assertGreater(python_score, 0.7)
        self.assertGreater(leadership_score, 0.7)


if __name__ == '__main__':
    unittest.main()