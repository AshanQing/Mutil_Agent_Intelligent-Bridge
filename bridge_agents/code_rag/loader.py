from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .schemas import CodeEntity, CodeLibrary, CodeLibraryMetadata, CodeLibraryType


_COLLECTION_TO_TYPE: Dict[str, CodeLibraryType] = {
    "rules": "rule",
    "formulas": "formula",
    "tables": "table",
    "variables": "variable",
}


def load_code_library(path: str | Path) -> CodeLibrary:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Code library must be a mapping: {source_path}")

    collection_key = _detect_collection_key(payload)
    library_type = _COLLECTION_TO_TYPE[collection_key]
    metadata_payload = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata = CodeLibraryMetadata(
        standard_code=str(metadata_payload.get("standard") or ""),
        standard_name=str(metadata_payload.get("standard_name") or ""),
        version=str(metadata_payload.get("version") or ""),
        library_type=str(metadata_payload.get("library_type") or collection_key),
        item_count=_as_optional_int(metadata_payload.get("item_count")),
        source_path=str(source_path),
        raw_payload=metadata_payload,
    )

    raw_items = payload.get(collection_key) or []
    if not isinstance(raw_items, list):
        raise ValueError(f"Code library collection must be a list: {collection_key} in {source_path}")

    entities = [
        _normalize_entity(item, library_type=library_type, metadata=metadata)
        for item in raw_items
        if isinstance(item, dict)
    ]
    return CodeLibrary(metadata=metadata, entities=entities)


def _detect_collection_key(payload: Dict[str, Any]) -> str:
    keys = [key for key in _COLLECTION_TO_TYPE if key in payload]
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one code collection key, got: {keys}")
    return keys[0]


def _normalize_entity(
    item: Dict[str, Any],
    *,
    library_type: CodeLibraryType,
    metadata: CodeLibraryMetadata,
) -> CodeEntity:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    applicability = _extract_applicability(item, library_type)
    execution = _extract_execution(item, library_type)
    related_entities = _extract_related_entities(item, meta)

    return CodeEntity(
        entity_id=str(item.get("id") or ""),
        library_type=library_type,
        standard_code=str(meta.get("standard") or meta.get("standard_code") or metadata.standard_code),
        standard_name=str(meta.get("standard_name") or metadata.standard_name),
        version=metadata.version,
        chapter=str(meta.get("chapter") or ""),
        clause=str(meta.get("clause") or ""),
        name=str(meta.get("name") or meta.get("title") or item.get("name") or ""),
        source=_extract_source(item, meta),
        source_file=metadata.source_path,
        original_text=_extract_original_text(item, meta),
        description=_extract_description(item, meta),
        applicability=applicability,
        execution=execution,
        related_entities=related_entities,
        raw_payload=item,
    )


def _extract_applicability(item: Dict[str, Any], library_type: CodeLibraryType) -> Dict[str, Any]:
    if library_type == "rule":
        value = item.get("applicability")
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        condition = item.get("condition") if isinstance(item.get("condition"), dict) else {}
        result = value if isinstance(value, dict) else {}
        if meta.get("applicability"):
            result = {**result, "scope": meta.get("applicability")}
        if condition:
            result = {
                **result,
                "condition_description": condition.get("description") or "",
                "condition_logic": condition.get("logic") or "",
            }
        return result
    if library_type == "formula":
        constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
        return {
            "materials": constraints.get("applicable_material") or [],
            "member_types": constraints.get("applicable_member") or [],
            "limit_states": constraints.get("limit_state") or [],
            "note": constraints.get("note") or "",
        }
    if library_type == "table":
        constraints = item.get("constraints") if isinstance(item.get("constraints"), dict) else {}
        schema = item.get("schema") if isinstance(item.get("schema"), dict) else {}
        return {
            "matching_method": constraints.get("matching_method") or "",
            "interpolation": constraints.get("interpolation"),
            "out_of_bounds": constraints.get("out_of_bounds") or "",
            "inputs": schema.get("inputs") or [],
            "outputs": schema.get("outputs") or [],
        }
    return {
        "roles": item.get("roles") or [],
        "unit": item.get("unit") or "",
        "value_type": item.get("value_type") or "",
    }


def _extract_execution(item: Dict[str, Any], library_type: CodeLibraryType) -> Dict[str, Any]:
    if library_type == "rule":
        value = item.get("action")
        return value if isinstance(value, dict) else {}
    if library_type == "formula":
        value = item.get("execution")
        math = item.get("math") if isinstance(item.get("math"), dict) else {}
        result = value if isinstance(value, dict) else {}
        if math:
            result = {
                **result,
                "expression": math.get("expression") or "",
                "variables": math.get("variables") or [],
                "output": math.get("output") or "",
            }
        return result
    if library_type == "table":
        data = item.get("data") if isinstance(item.get("data"), list) else []
        return {"row_count": len(data)}
    return {
        "symbol": item.get("symbol") or "",
        "definition": item.get("definition") or item.get("description") or "",
    }


def _extract_related_entities(item: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, List[str]]:
    related: Dict[str, List[str]] = {}
    related_refs = meta.get("related_refs") if isinstance(meta.get("related_refs"), dict) else {}
    item_related_refs = item.get("related_refs") if isinstance(item.get("related_refs"), dict) else {}

    _extend(related, "rules", related_refs.get("rules"))
    _extend(related, "tables", related_refs.get("tables"))
    _extend(related, "formulas", related_refs.get("formulas"))
    _extend(related, "variables", related_refs.get("variables"))
    _extend(related, "rules", item_related_refs.get("rules"))
    _extend(related, "tables", item_related_refs.get("tables"))
    _extend(related, "formulas", item_related_refs.get("formulas"))
    _extend(related, "variables", item_related_refs.get("variables"))
    _extend(related, "rules", meta.get("related_rules"))
    _extend(related, "tables", meta.get("related_tables"))
    _extend(related, "formulas", meta.get("related_formulas"))
    _extend(related, "variables", meta.get("related_variables"))
    _extend(related, "rules", item.get("related_rules"))
    _extend(related, "tables", item.get("related_tables"))
    _extend(related, "formulas", item.get("related_formulas"))
    _extend(related, "variables", item.get("related_variables"))

    math = item.get("math") if isinstance(item.get("math"), dict) else {}
    _extend(related, "variables", math.get("variables"))
    _extend(related, "variables", [math.get("output")] if math.get("output") else [])

    return {key: values for key, values in related.items() if values}


def _extract_source(item: Dict[str, Any], meta: Dict[str, Any]) -> str:
    if meta.get("source"):
        return str(meta.get("source"))
    source_files = item.get("_source_files")
    if isinstance(source_files, list) and source_files:
        return ", ".join(str(value) for value in source_files)
    if item.get("_source_file"):
        return str(item.get("_source_file"))
    return ""


def _extract_original_text(item: Dict[str, Any], meta: Dict[str, Any]) -> str:
    if meta.get("original_text"):
        return str(meta.get("original_text"))
    requirement = item.get("requirement") if isinstance(item.get("requirement"), dict) else {}
    return str(requirement.get("expression") or requirement.get("description") or item.get("original_text") or "")


def _extract_description(item: Dict[str, Any], meta: Dict[str, Any]) -> str:
    if meta.get("description"):
        return str(meta.get("description"))
    requirement = item.get("requirement") if isinstance(item.get("requirement"), dict) else {}
    condition = item.get("condition") if isinstance(item.get("condition"), dict) else {}
    return str(
        item.get("description")
        or requirement.get("description")
        or condition.get("description")
        or ""
    )


def _extend(target: Dict[str, List[str]], key: str, values: Any) -> None:
    if values in [None, "", [], {}]:
        return
    bucket = target.setdefault(key, [])
    for value in _flatten_values(values):
        text = str(value)
        if text and text not in bucket:
            bucket.append(text)


def _flatten_values(values: Any) -> Iterable[Any]:
    if values in [None, "", [], {}]:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        flattened: list[Any] = []
        for item in values:
            flattened.extend(_flatten_values(item))
        return flattened
    return []


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
