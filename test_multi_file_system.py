#!/usr/bin/env python
"""Test script for multi-file upload and multi-sheet analysis"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test server health"""
    print("🔍 Testing server health...")
    try:
        resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Server running - {data['sessions']} active sessions")
            return True
        else:
            print(f"❌ Server error: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_upload_multiple_files():
    """Test uploading multiple files"""
    print("\n📤 Testing multi-file upload...")

    # Check for sample files
    sample_dir = Path("data/samples")
    files_to_upload = []

    # Look for Excel files
    for ext in ['*.xlsx', '*.xls']:
        for file_path in sample_dir.glob(ext):
            files_to_upload.append(file_path)

    # Look for CSV files
    for file_path in sample_dir.glob('*.csv'):
        files_to_upload.append(file_path)

    if not files_to_upload:
        print("❌ No sample files found in data/samples/")
        return None

    print(f"📁 Found {len(files_to_upload)} files to upload:")
    for f in files_to_upload:
        print(f"   - {f.name}")

    # Prepare multipart form data
    files = []
    for file_path in files_to_upload:
        files.append(('files', (file_path.name, open(file_path, 'rb'), 'application/octet-stream')))

    try:
        resp = requests.post(f"{BASE_URL}/api/upload", files=files, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            session_id = data['session_id']
            print(f"✅ Upload successful!")
            print(f"   Session ID: {session_id}")
            print(f"   Files: {data.get('filenames', [])}")
            print(f"   Sheets: {len(data.get('sheet_names', []))} total")
            if data.get('sheets_context'):
                print(f"   Context: {data['sheets_context'][:100]}...")
            return session_id
        else:
            print(f"❌ Upload failed: {resp.status_code} - {resp.text}")
            return None

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def test_get_sheets(session_id):
    """Test getting sheet information"""
    print(f"\n📊 Testing sheet information for session {session_id}...")

    try:
        resp = requests.get(f"{BASE_URL}/api/sheets/{session_id}", timeout=10)

        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Retrieved sheet info!")
            print(f"   Files: {data.get('files', [])}")
            print(f"   Sheets: {len(data.get('sheets', []))}")

            for sheet in data.get('sheets', [])[:3]:  # Show first 3
                file_info = f" ({sheet['file_name']})" if sheet.get('file_name') else ""
                print(f"   - {sheet['name']}{file_info}: {sheet['rows']} rows × {sheet['columns']} cols")

            if data.get('relationships'):
                print(f"   Relationships: {len(data['relationships'])} detected")

            return data
        else:
            print(f"❌ Get sheets failed: {resp.status_code} - {resp.text}")
            return None

    except Exception as e:
        print(f"❌ Get sheets error: {e}")
        return None

def test_analyze(session_id):
    """Test analysis with question"""
    print(f"\n🧠 Testing analysis for session {session_id}...")

    payload = {
        "session_id": session_id,
        "question": "tóm tắt dữ liệu và phân tích xu hướng chính"
    }

    try:
        resp = requests.post(f"{BASE_URL}/api/analyze", json=payload, timeout=30)

        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Analysis successful!")
            print(f"   Answer: {data['answer'][:200]}...")
            print(f"   Charts: {len(data.get('charts', []))}")
            print(f"   Report ID: {data.get('report_id')}")
            return data
        else:
            print(f"❌ Analysis failed: {resp.status_code} - {resp.text}")
            return None

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return None

def main():
    """Run all tests"""
    print("🚀 Testing Multi-File & Multi-Sheet Data Analysis System\n")

    # Test 1: Health check
    if not test_health():
        print("\n❌ Server not running. Start with: python -m uvicorn backend.app.main:app --reload --port 8000")
        return

    # Test 2: Upload multiple files
    session_id = test_upload_multiple_files()
    if not session_id:
        print("\n❌ Upload failed. Check if sample files exist in data/samples/")
        return

    # Test 3: Get sheet information
    sheets_data = test_get_sheets(session_id)
    if not sheets_data:
        return

    # Test 4: Analyze data
    analysis_data = test_analyze(session_id)
    if not analysis_data:
        return

    print("\n🎉 All tests passed! System is working correctly.")
    print(f"\n📋 Session ID for manual testing: {session_id}")
    print(f"\n🌐 API Docs: http://localhost:8000/docs")
    print(f"📊 Frontend: http://localhost:8000")

if __name__ == "__main__":
    main()