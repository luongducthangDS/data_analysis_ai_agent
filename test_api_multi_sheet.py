#!/usr/bin/env python
"""Test API endpoints for multi-sheet functionality"""

import sys
import requests
import json
from pathlib import Path

# Test file
TEST_FILE = "data/samples/test_multi_sheet.xlsx"

BASE_URL = "http://localhost:8000"

print("🚀 Testing Multi-Sheet API Endpoints\n")

# Test 1: Health check
print("1️⃣  Health Check...")
try:
    resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
    if resp.status_code == 200:
        print("   ✓ Server is running")
    else:
        print("   ⚠️  Server not responding properly")
except requests.exceptions.ConnectionError:
    print("   ❌ Server is not running. Start it with: python -m uvicorn backend.app.main:app --reload")
    sys.exit(1)

# Test 2: Upload multi-sheet Excel
print("\n2️⃣  Upload Multi-Sheet Excel...")
with open(TEST_FILE, "rb") as f:
    files = [("files", (Path(TEST_FILE).name, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))]
    resp = requests.post(f"{BASE_URL}/api/upload", files=files, timeout=10)

if resp.status_code == 200:
    data = resp.json()
    session_id = data["session_id"]
    print(f"   ✓ Upload successful: {session_id}")
    print(f"   - Filename: {data['filename']}")
    print(f"   - Sheets: {data.get('sheet_names')}")
    if data.get("sheets_context"):
        print(f"   - Sheets detected: {len(data.get('sheet_names', []))} sheets")
else:
    print(f"   ❌ Upload failed: {resp.text}")
    sys.exit(1)

# Test 3: Get sheets info
print("\n3️⃣  Get Sheets Information...")
resp = requests.get(f"{BASE_URL}/api/sheets/{session_id}", timeout=10)
if resp.status_code == 200:
    data = resp.json()
    print(f"   ✓ Found {len(data['sheets'])} sheets:")
    for sheet in data["sheets"]:
        print(f"   - {sheet['name']}: {sheet['rows']} rows × {sheet['columns']} columns")
        print(f"     Columns: {', '.join(sheet['column_names'][:5])}")
    
    if data["relationships"]:
        print(f"\n   Relationships detected:")
        for rel in data["relationships"]:
            print(f"   - {rel['sheet1']} ↔ {rel['sheet2']} ({rel['relationship_type']})")
else:
    print(f"   ⚠️  Get sheets failed: {resp.text}")

# Test 4: Merge sheets
print("\n4️⃣  Merge Sheets...")
merge_payload = {
    "session_id": session_id,
    "sheet_names": ["Products", "Sales"],
    "join_key": "product_id"
}
resp = requests.post(f"{BASE_URL}/api/merge-sheets", json=merge_payload, timeout=10)
if resp.status_code == 200:
    data = resp.json()
    print(f"   ✓ Merge successful: {data['merged_sheet_name']}")
    print(f"   - Merged: {data['merged_rows']} rows × {data['merged_columns']} columns")
else:
    print(f"   ⚠️  Merge failed: {resp.text}")

print("\n✅ API tests completed!")
print(f"\nSession ID for debugging: {session_id}")
