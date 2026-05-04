#!/usr/bin/env python
"""Test multi-sheet functionality"""

import pandas as pd
from backend.app.services.multi_sheet_analyzer import MultiSheetAnalyzer

# Create test data
df1 = pd.DataFrame({
    "product_id": [1, 2, 3, 4, 5],
    "product_name": ["A", "B", "C", "D", "E"],
    "category": ["Cat1", "Cat1", "Cat2", "Cat2", "Cat3"]
})

df2 = pd.DataFrame({
    "product_id": [1, 2, 3, 4, 5],
    "sales": [100, 200, 150, 300, 250],
    "revenue": [1000, 2000, 1500, 3000, 2500]
})

df3 = pd.DataFrame({
    "product_id": [1, 2, 3, 4, 5],
    "stock": [50, 100, 75, 200, 150],
    "warehouse": ["A", "B", "A", "C", "B"]
})

sheets = {
    "Products": df1,
    "Sales": df2,
    "Inventory": df3
}

print("📊 Testing MultiSheetAnalyzer\n")

# Test 1: Analyze metadata
print("1️⃣  Analyzing sheet metadata...")
metadata = MultiSheetAnalyzer.analyze_sheets_metadata(sheets)
for sheet_name, info in metadata.items():
    print(f"   {sheet_name}: {info.rows} rows × {info.columns} columns")
    print(f"   Columns: {', '.join(info.column_names)}")

# Test 2: Detect relationships
print("\n2️⃣  Detecting relationships between sheets...")
relationships = MultiSheetAnalyzer.detect_relationships(sheets)
for rel in relationships:
    print(f"   {rel.sheet1} ↔ {rel.sheet2}")
    print(f"   - Type: {rel.relationship_type}")
    print(f"   - Join Key: {rel.join_key}")
    print(f"   - Similarity: {rel.similarity_score}")

# Test 3: Generate context
print("\n3️⃣  Generating sheets context...")
context = MultiSheetAnalyzer.generate_sheets_context(sheets, relationships)
print(context)

# Test 4: Merge sheets
print("\n4️⃣  Merging related sheets...")
try:
    merged = MultiSheetAnalyzer.merge_related_sheets(sheets, relationships)
    print(f"   Original sheets: {list(sheets.keys())}")
    print(f"   After merge: {list(merged.keys())}")
    for name, df in merged.items():
        if isinstance(df, pd.DataFrame) and ("merged" in name or "with" in name):
            print(f"   ✓ {name}: {len(df)} rows × {len(df.columns)} columns")
except Exception as e:
    print(f"   ⚠️  Merge failed: {e}")

print("\n✅ Multi-sheet analyzer tests completed!")
