from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SheetInfo:
    """Metadata about a sheet."""

    name: str
    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]


@dataclass
class SheetRelationship:
    """Detected relationship between two sheets."""

    sheet1: str
    sheet2: str
    join_key: str | None
    relationship_type: str
    similarity_score: float


class MultiSheetAnalyzer:
    """Analyzes relationships between multiple sheets and merges them intelligently."""

    @staticmethod
    def read_all_sheets(file_path: str) -> dict[str, pd.DataFrame]:
        """Read all sheets from an Excel file with smart header detection."""
        try:
            raw = pd.read_excel(file_path, sheet_name=None)
            result = {}
            for sheet_name, df in raw.items():
                result[sheet_name] = MultiSheetAnalyzer._fix_header(file_path, sheet_name, df)
            return result
        except Exception as exc:
            raise ValueError(f"Error reading Excel sheets: {exc}") from exc

    @staticmethod
    def _fix_header(file_path: str, sheet_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Re-read with header=N when many columns are Unnamed."""
        if df.empty or len(df.columns) == 0:
            return df

        unnamed_ratio = sum(
            1 for col in df.columns if str(col).startswith("Unnamed:") or str(col).startswith("[")
        ) / len(df.columns)
        if unnamed_ratio < 0.5:
            return MultiSheetAnalyzer._coerce_numeric(df)

        for skip in range(1, 5):
            try:
                candidate = pd.read_excel(file_path, sheet_name=sheet_name, header=skip)
                if candidate.empty:
                    continue
                new_unnamed = sum(
                    1 for col in candidate.columns if str(col).startswith("Unnamed:")
                ) / len(candidate.columns)
                if new_unnamed < unnamed_ratio:
                    return MultiSheetAnalyzer._coerce_numeric(candidate)
            except Exception:
                continue

        return MultiSheetAnalyzer._coerce_numeric(df)

    @staticmethod
    def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """Try to coerce object columns that look numeric to float."""
        result = df.copy()
        for col in result.select_dtypes(include="object").columns:
            converted = pd.to_numeric(result[col], errors="coerce")
            if converted.notna().sum() / max(len(result), 1) >= 0.7:
                result[col] = converted
        return result

    @staticmethod
    def analyze_sheets_metadata(sheets: dict[str, pd.DataFrame]) -> dict[str, SheetInfo]:
        """Extract metadata from all sheets."""
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
    def detect_relationships(sheets: dict[str, pd.DataFrame]) -> list[SheetRelationship]:
        """Detect simple relationships between sheets."""
        relationships = []
        sheet_names = list(sheets.keys())

        for index, sheet1_name in enumerate(sheet_names):
            for sheet2_name in sheet_names[index + 1 :]:
                sheet1 = sheets[sheet1_name]
                sheet2 = sheets[sheet2_name]
                common_cols = set(sheet1.columns) & set(sheet2.columns)
                join_key = None

                if common_cols:
                    id_candidates = [col for col in common_cols if "id" in str(col).lower()]
                    join_key = id_candidates[0] if id_candidates else list(common_cols)[0]

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
        """Merge related sheets based on detected relationships."""
        merged = dict(sheets)
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
                    merged_df = pd.merge(
                        df1,
                        df2,
                        on=rel.join_key,
                        how="left",
                        suffixes=("_x", "_y"),
                    )
                    merged[f"{rel.sheet1}_merged"] = merged_df
                elif rel.relationship_type == "parent_child":
                    merged_df = pd.merge(
                        df1,
                        df2,
                        on=rel.join_key,
                        how="left",
                        suffixes=("", "_child"),
                    )
                    merged[f"{rel.sheet1}_with_{rel.sheet2}"] = merged_df
            except Exception:
                continue

        return merged

    @staticmethod
    def create_summary_dataframe(
        sheets: dict[str, pd.DataFrame],
        relationships: list[SheetRelationship],
    ) -> pd.DataFrame:
        """Create a summary dataframe about all sheets for AI analysis."""
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
                    "column_list": ", ".join(map(str, df.columns[:10])),
                    "numeric_cols_list": ", ".join(map(str, numeric_cols[:5])) or "None",
                    "categorical_cols_list": ", ".join(map(str, categorical_cols[:5])) or "None",
                    "relationships": len(
                        [
                            rel
                            for rel in relationships
                            if rel.sheet1 == sheet_name or rel.sheet2 == sheet_name
                        ]
                    ),
                }
            )

        return pd.DataFrame(summary_data)

    @staticmethod
    def generate_sheets_context(
        sheets: dict[str, pd.DataFrame],
        relationships: list[SheetRelationship],
    ) -> str:
        """Generate a text description of the file structure for LLM analysis."""
        lines = [
            "**Excel File Structure Analysis**\n",
            f"Total Sheets: {len(sheets)}\n",
        ]

        lines.append("\n**Sheets Overview:**")
        for sheet_name, df in sheets.items():
            lines.append(f"\n- **{sheet_name}**: {len(df)} rows x {len(df.columns)} columns")
            columns = ", ".join(map(str, df.columns.tolist()[:8]))
            lines.append(f"  - Columns: {columns}")
            if len(df.columns) > 8:
                lines.append(f"    ... and {len(df.columns) - 8} more")

        if relationships:
            lines.append("\n**Detected Relationships:**")
            for rel in relationships:
                if rel.similarity_score > 0.3:
                    join_str = f" (join key: {rel.join_key})" if rel.join_key else ""
                    lines.append(
                        f"- {rel.sheet1} <-> {rel.sheet2}: {rel.relationship_type}{join_str}"
                    )

        return "\n".join(lines)
