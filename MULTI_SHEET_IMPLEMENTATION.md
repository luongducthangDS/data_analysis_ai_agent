# Multi-Sheet Excel Support Implementation Summary

## What's Been Implemented

Your data analysis AI agent now supports **intelligent multi-sheet Excel analysis**. The system can:

✅ **Detect sheet relationships** automatically
- Finds common columns between sheets
- Identifies join keys (e.g., product_id, customer_id)
- Classifies relationships: parent-child, sibling, independent

✅ **Analyze sheet structure**
- Extracts metadata about each sheet (rows, columns, types)
- Generates contextual descriptions for AI analysis
- Provides similarity scores between related sheets

✅ **Merge sheets intelligently**
- Joins sheets on common keys
- Preserves data integrity with proper left/right joins
- Handles multiple sheet merges systematically

✅ **New API Endpoints**
- `GET /api/sheets/{session_id}` - Get all sheets and their relationships
- `POST /api/merge-sheets` - Merge multiple sheets on specified join key

## Key Files Modified

### 1. **New Service: `multi_sheet_analyzer.py`**
   - Core logic for analyzing and merging multiple sheets
   - Contains:
     - `SheetInfo` & `SheetRelationship` data classes
     - `MultiSheetAnalyzer` class with methods for:
       - Reading all sheets from Excel
       - Analyzing metadata
       - Detecting relationships
       - Merging sheets
       - Generating context for LLM

### 2. **Updated: `storage.py`**
   - Added fields to `DatasetSession`:
     - `sheets`: dict of all sheets
     - `sheet_relationships`: list of detected relationships
     - `sheets_context`: textual description of sheet structure
   - Updated `SessionStore.create()` to read and analyze sheets
   - New method `_read_dataframe_with_sheets()` for multi-sheet handling

### 3. **Updated: `schemas.py`**
   - Enhanced `UploadResponse` with:
     - `sheets`: list of sheet names
     - `sheets_context`: description of sheet structure
   - New models:
     - `SheetData` - individual sheet metadata
     - `GetSheetsResponse` - response for sheets endpoint
     - `MergeSheetsRequest` - request to merge sheets
     - `MergeSheetsResponse` - merge result

### 4. **Updated: `main.py`**
   - New endpoint: `GET /api/sheets/{session_id}`
     - Returns all sheets with their metadata and relationships
   - New endpoint: `POST /api/merge-sheets`
     - Merges specified sheets on a join key
   - Updated `/api/upload` to include sheets info in response

### 5. **Updated: `README.md`**
   - Added section on multi-sheet support
   - Documented new API endpoints with examples
   - Added usage instructions

## How It Works

### Upload Flow
```
User uploads Excel with 3 sheets
         ↓
SessionStore reads all sheets
         ↓
MultiSheetAnalyzer analyzes structure
         ↓
Detects relationships between sheets
         ↓
API response includes sheets info
```

### Analysis Flow
```
User requests sheet info
         ↓
GET /api/sheets/{session_id}
         ↓
Returns all sheets + relationships
         ↓
User can choose to merge sheets
         ↓
POST /api/merge-sheets with join key
         ↓
Sheets merged and available for analysis
```

## Example Usage

### 1. Upload Excel File
```bash
curl -F "file=@sales_data.xlsx" http://localhost:8000/api/upload
```

Response includes detected sheets:
```json
{
  "session_id": "abc123",
  "sheets": ["Products", "Sales", "Inventory"],
  "sheets_context": "📊 **Excel File Structure Analysis**\nTotal Sheets: 3\n..."
}
```

### 2. Get Sheet Information
```bash
curl http://localhost:8000/api/sheets/abc123
```

Returns all sheets with metadata and detected relationships.

### 3. Merge Sheets
```bash
curl -X POST http://localhost:8000/api/merge-sheets \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123",
    "sheet_names": ["Products", "Sales"],
    "join_key": "product_id"
  }'
```

Creates merged dataset with combined data from both sheets.

## Testing

### Run Tests
```bash
# Test multi-sheet analyzer directly
python test_multi_sheet.py

# Test API endpoints (requires running server)
python -m uvicorn backend.app.main:app --reload
python test_api_multi_sheet.py
```

### Test Files
- `test_multi_sheet.py` - Direct unit tests for analyzer
- `test_api_multi_sheet.py` - Integration tests for API
- `data/samples/test_multi_sheet.xlsx` - Sample Excel with 3 sheets

## Relationship Detection Logic

The system identifies relationships by analyzing:

1. **Common columns** between sheets
2. **Row counts** to determine parent-child relationships
3. **Data patterns** to infer join keys

### Relationship Types
- **sibling**: Same row count, related data
- **parent_child**: One sheet has fewer rows (detail vs summary)
- **related**: Shares join key but different structures
- **independent**: No meaningful relationship

## Next Steps (Optional Enhancements)

1. **AI-Powered Analysis**: Use LLM to generate insights from merged data
2. **Auto Merge**: Automatically merge sheets for analysis (configurable)
3. **Hierarchy Visualization**: Show sheet relationships in UI
4. **Advanced Merging**: Support complex merges (multiple keys, outer joins)
5. **Sheet Selection UI**: Let users select which sheets to analyze

## Notes

- CSV files bypass multi-sheet analysis (single DataFrame)
- Sheets are analyzed on upload, relationships detected automatically
- Merge preserves all data with left join (non-matching rows retained)
- System handles up to 10MB Excel files
- Works with .xlsx and .xls formats
