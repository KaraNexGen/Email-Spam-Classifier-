"""
Simple Installation Script for ESIS
==================================
Installs only the essential packages needed for ESIS to work
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Main installation function"""
    print("🛡️ ESIS Simple Installation")
    print("=" * 40)
    
    # Essential packages
    essential_packages = [
        "pandas",
        "numpy", 
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "nltk",
        "textblob",
        "flask",
        "flask-cors",
        "joblib"
    ]
    
    print("Installing essential packages...")
    
    success_count = 0
    for package in essential_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(essential_packages)} packages")
    
    if success_count == len(essential_packages):
        print("\n🎉 Installation complete!")
        print("\nNext steps:")
        print("1. Run: python esis_simple.py")
        print("2. Or run: python simple_webapp.py")
        print("3. Access web app at: http://localhost:5000")
    else:
        print("\n⚠️ Some packages failed to install.")
        print("You may need to install them manually:")
        for package in essential_packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()
