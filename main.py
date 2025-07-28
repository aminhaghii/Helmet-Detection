#!/usr/bin/env python3
"""
HSE Vision - Professional Helmet Detection System
Main Application Entry Point

This is the main entry point for the HSE Vision helmet detection system.
It provides a simple interface to run the professional helmet detection.

Author: HSE Vision Team
Version: 2.0
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

def main():
    """Main application entry point"""
    print("=" * 60)
    print("🔒 HSE Vision - Professional Helmet Detection System v2.0")
    print("=" * 60)
    print()
    print("Available options:")
    print("1. Professional Helmet Detection (Recommended)")
    print("2. Simple Test")
    print("3. Camera Test")
    print("4. Exit")
    print()
    
    while True:
        try:
            choice = input("Select an option (1-4): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Professional Helmet Detection System...")
                from advanced_test import main as advanced_main
                advanced_main()
                break
                
            elif choice == "2":
                print("\n🔧 Starting Simple Test...")
                from simple_test import main as simple_main
                simple_main()
                break
                
            elif choice == "3":
                print("\n📹 Starting Camera Test...")
                from test_camera import main as camera_main
                camera_main()
                break
                
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()