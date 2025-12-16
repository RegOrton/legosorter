#!/usr/bin/env python3
"""
Simple test to verify LDView is working in the Docker container.
"""

import subprocess
import sys
import os

def test_ldview():
    """Test if LDView is available and working."""
    print("Testing LDView installation...")

    try:
        result = subprocess.run(
            ["ldview", "-v"],
            capture_output=True,
            timeout=5,
            text=True
        )

        version_output = result.stdout + result.stderr
        print(f"✓ LDView is installed:")
        print(f"  {version_output.strip()}")
        return True
    except FileNotFoundError:
        print("✗ LDView not found in PATH")
        return False
    except Exception as e:
        print(f"✗ Error testing LDView: {e}")
        return False


def test_ldraw_library():
    """Test if LDraw library is accessible."""
    print("\nTesting LDraw library...")

    ldraw_dir = os.environ.get('LDRAWDIR')
    print(f"LDRAWDIR environment variable: {ldraw_dir}")

    if ldraw_dir and os.path.exists(ldraw_dir):
        print(f"✓ LDraw directory exists: {ldraw_dir}")

        # Count .dat files
        parts_dir = os.path.join(ldraw_dir, "parts")
        if os.path.exists(parts_dir):
            dat_files = len([f for f in os.listdir(parts_dir) if f.endswith('.dat')])
            print(f"  Found {dat_files} .dat files in parts directory")
        return True
    else:
        print(f"✗ LDraw directory not found: {ldraw_dir}")
        return False


def test_renderer_import():
    """Test if we can import the LDViewRenderer."""
    print("\nTesting Python renderer import...")

    try:
        from ldview_renderer import LDViewRenderer
        print("✓ Successfully imported LDViewRenderer")
        return True
    except ImportError as e:
        print(f"✗ Failed to import LDViewRenderer: {e}")
        return False
    except Exception as e:
        print(f"✗ Error importing: {e}")
        return False


def main():
    print("=" * 60)
    print("LDView Container Test")
    print("=" * 60)

    tests = [
        ("LDView Installation", test_ldview),
        ("LDraw Library", test_ldraw_library),
        ("Renderer Import", test_renderer_import),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r for _, r in results)
    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! LDView is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
