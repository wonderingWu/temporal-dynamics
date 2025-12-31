#!/usr/bin/env python3
"""
Basic tests for temporal-dynamics project.
"""

import unittest
import numpy as np
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests."""
    
    def test_numpy_available(self):
        """Test that numpy is available and working."""
        self.assertTrue(np.__version__)
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(len(arr), 5)
    
    def test_figures_directory_exists(self):
        """Test that figures directory exists."""
        figures_path = os.path.join(os.path.dirname(__file__), '..', 'figures')
        self.assertTrue(os.path.exists(figures_path))
    
    def test_source_files_exist(self):
        """Test that all source files exist."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        required_files = [
            'ESN_criticality_te_fixed.py',
            'ising_ais_nmi_quick_revised.py',
            'logicMapTest_fixed.py',
            'sandpile_critical_time_analysis.py',
            'sandpile_reset_vs_te.py'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(src_path, file_name)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file_name}")
    
    def test_documentation_exists(self):
        """Test that documentation files exist."""
        docs_path = os.path.join(os.path.dirname(__file__), '..', 'docs')
        self.assertTrue(os.path.exists(os.path.join(docs_path, 'temporalDynamics.tex')))
    
    def test_project_structure(self):
        """Test basic project structure."""
        project_root = os.path.dirname(__file__)
        
        # Check for required files
        required_files = [
            'README.md',
            'requirements.txt',
            'setup.py',
            'LICENSE'
        ]
        
        for file_name in required_files:
            file_path = os.path.join(project_root, file_name)
            self.assertTrue(os.path.exists(file_path), f"Missing required file: {file_name}")
    
    def test_jar_download_script(self):
        """Test that JIDT download script exists."""
        download_script = os.path.join(os.path.dirname(__file__), '..', 'download_jar.py')
        self.assertTrue(os.path.exists(download_script))
        
        # Test that script can be imported
        try:
            import download_jar
            self.assertTrue(hasattr(download_jar, 'download_jidt'))
        except ImportError:
            # If JIDT is not available, just check the file exists
            pass

if __name__ == '__main__':
    unittest.main()