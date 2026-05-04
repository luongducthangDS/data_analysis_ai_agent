from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class SheetInfo:
    """Metadata about a sheet"""
    name: str
    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]


@dataclass
class SheetRelationship:
    """Detected relationship between two sheets"""
    sheet1: str
    sheet2: str
    join_key: str | None
    relationship_type: str  # "parent_child", "sibling", "related", "independent"
    similarity_score: float


class MultiSheetAnalyzer:
    """Analyzes relationships between multiple sheets and merges them intelligently"""

    @staticmethod
    def read_all_sheets(file_path: str) -> dict[str, pd.DataFrame]:
        """Read all sheets from Excel file"""
        try:
            sheets = pd.read_excel(file_path, sheet_name=None)
            return sheets
        except Exception as exc:
            raise ValueError(f"Error reading Excel sheets: {str(exc)}") from exc

    @staticmethod
    def analyze_sheets_metadata(sheets: dict[str, pd.DataFrame]) -> dict[str, SheetInfo]:
        """Extract metadata from all sheets"""
        metadata = {}
        for sheet_name, df in sheets.items():
            metadata[sheet_name] = SheetInfo(
                name=sheet_name,
                rows=len(df),
                columns=len(df.columns),
                column_names=df.columns.tolist(),
                column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            )
        return metadata

    @staticmethod
    def detect_relationships(
        sheets: dict[str, pd.DataFrame],
    ) -> list[SheetRelationship]:
        """Detect relationships between sheets"""
        relationships = []
        sheet_names = list(sheets.keys())

        for i, sheet1_name in enumerate(sheet_names):
            for sheet2_name in sheet_names[i + 1 :]:
                sheet1 = sheets[sheet1_name]
                sheet2 = sheets[sheet2_name]

                # Find common columns (potential join keys)
                common_cols = set(sheet1.columns) & set(sheet2.columns)
                join_key = None

                if common_cols:
                    # Prefer ID-like columns
                    id_candidates = [col for col in common_cols if "id" in col.lower()]
                    if id_candidates:
                        join_key = id_candidates[0]
                    else:
                        join_key = list(common_cols)[0]

                # Determine relationship type
                if join_key and len(sheet1) == len(sheet2):
                    relationship_type = "sibling"
                    similarity = 0.8
                elif join_key and len(sheet2) < len(sheet1):
                    relationship_type = "parent_child"
                    similarity = 0.7
                elif join_key:
                    relationship_type = "related"
                    similarity = 0.6
                else:
                    relationship_type = "independent"
                    similarity = 0.0

                relationships.append(
                    SheetRelationship(
                        sheet1=sheet1_name,
                        sheet2=sheet2_name,
                        join_key=join_key,
                        relationship_type=relationship_type,
                        similarity_score=similarity,
                    )
                )

        return relationships

    @staticmethod
    def merge_related_sheets(
        sheets: dict[str, pd.DataFrame],
        relationships: list[SheetRelationship],
        threshold: float = 0.5,
    ) -> dict[str, pd.DataFrame]:
        """
        Merge related sheets based on detected relationships.
        Returns a dict of merged dataframes.
        """
        merged = dict(sheets)  # Start with original sheets
        processed_pairs = set()

        for rel in relationships:
            if rel.similarity_score < threshold or rel.join_key is None:
                continue

            pair = tuple(sorted([rel.sheet1, rel.sheet2]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            df1 = merged.get(rel.sheet1)
            df2 = merged.get(rel.sheet2)

            if df1 is None or df2 is None:
                continue

            try:
                if rel.relationship_type == "sibling":
                    # Merge columns (left merge)
                    merged_df = pd.merge(
                        df1, df2, on=rel.join_key, how="left", suffixes=("_x", "_y")
                    )
                    merged[f"{rel.sheet1}_merged"] = merged_df
                    # Mark originals for potential removal
                    merged[f"{rel.sheet1}_merged__original"] = True

                elif rel.relationship_type == "parent_child":
                    # Join parent with child
                    merged_df = pd.merge(
                        df1, df2, on=rel.join_key, how="left", suffixes=("", "_child")
                    )
                    merged[f"{rel.sheet1}_with_{rel.sheet2}"] = merged_df
                    merged[f"{rel.sheet1}_with_{rel.sheet2}__original"] = True

            except Exception as exc:
                # Silently skip merge if it fails
                pass

        return merged

    @staticmethod
    def create_summary_dataframe(
        sheets: dict[str, pd.DataFrame],
        relationships: list[SheetRelationship],
    ) -> pd.DataFrame:
        """
        Create a summary dataframe about all sheets for AI analysis.
        """
        summary_data = []

        for sheet_name, df in sheets.items():
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            summary_data.append(
                {
                    "sheet_name": sheet_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric_cols": len(numeric_cols),
                    "categorical_cols": len(categorical_cols),
                    "column_list": ", ".join(df.columns[:10]),
                    "numeric_cols_list": ", ".join(numeric_cols[:5]) or "None",
                    "categorical_cols_list": ", ".join(categorical_cols[:5]) or "None",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        return summary_df

    @staticmethod
    def generate_sheets_context(
        sheets: dict[str, pd.DataFrame],
        relationships: list[SheetRelationship],
    ) -> str:
        """
        Generate a text description of the file structure for LLM analysis.
        """
        lines = [
            "📊 **Excel File Structure Analysis**\n",
            f"Total Sheets: {len(sheets)}\n",
        ]

        # Sheet details
        lines.append("\n**Sheets Overview:**")
        for sheet_name, df in sheets.items():
            lines.append(f"\n- **{sheet_name}**: {len(df)} rows × {len(df.columns)} columns")
            lines.append(f"  - Columns: {', '.join(df.columns.tolist()[:8])}")
            if len(df.columns) > 8:
                lines.append(f"    ... and {len(df.columns) - 8} more")

        # Relationships
        if relationships:
            lines.append("\n**Detected Relationships:**")
            for rel in relationships:
                if rel.similarity_score > 0.3:
                    join_str = f" (join key: {rel.join_key})" if rel.join_key else ""
                    lines.append(
                        f"- {rel.sheet1} ↔ {rel.sheet2}: {rel.relationship_type}{join_str}"
                    )

        return "\n".join(lines)
