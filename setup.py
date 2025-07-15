#!/usr/bin/env python3
"""
Tennis Intelligence System - Optimization Setup Script
=====================================================

This script helps install and configure the performance optimizations
for the tennis intelligence system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and show progress."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but you have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Core dependencies (required)
    core_deps = [
        "scikit-learn>=1.3.2",
        "numpy>=1.24.3",
        "pandas>=2.0.3"
    ]
    
    for dep in core_deps:
        if not run_command(f"pip install \"{dep}\"", f"Installing {dep}"):
            return False
    
    # Optional dependencies (visualization)
    optional_deps = [
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2"
    ]
    
    print("📊 Installing optional visualization dependencies...")
    for dep in optional_deps:
        run_command(f"pip install \"{dep}\"", f"Installing {dep}")
    
    return True

def verify_installation():
    """Verify that key components can be imported."""
    print("🔍 Verifying installation...")
    
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
    except ImportError:
        print("❌ scikit-learn import failed")
        return False
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError:
        print("❌ numpy import failed")
        return False
    
    try:
        import pandas
        print("✅ pandas imported successfully")
    except ImportError:
        print("❌ pandas import failed")
        return False
    
    return True

def setup_directories():
    """Create necessary directories for optimized system."""
    print("📁 Setting up directories...")
    
    directories = [
        "tennis_data",
        "logs",
        "cache"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   • Created/verified: {dir_name}/")
    
    return True

def create_config_backup():
    """Create backup of existing configuration."""
    config_file = Path("src/config/settings.py")
    if config_file.exists():
        backup_file = Path("src/config/settings.py.backup")
        if not backup_file.exists():
            import shutil
            shutil.copy2(config_file, backup_file)
            print("✅ Created backup of existing configuration")
    return True


def main():
    """Main setup function."""
    print("🎾 Tennis Intelligence System - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        print("❌ Directory setup failed")
        sys.exit(1)
    
    # Create config backup
    create_config_backup()
    

if __name__ == "__main__":
    main() 