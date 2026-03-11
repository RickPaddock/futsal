"""
File I/O utilities for JSON artifacts.

Per CLAUDE.md Section 7 (Utilities):
- Save/load JSON with validation
- Atomic writes (temp file + rename)
- Pretty-printed output for human readability
- JSON is source of truth
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

# Optional jsonschema support
try:
    from jsonschema import validate, ValidationError as JSONSchemaValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    JSONSchemaValidationError = Exception
    validate = None

# For Pydantic model support
try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError
except ImportError:
    BaseModel = None
    PydanticValidationError = Exception

T = TypeVar('T')


def save_json(
    data: Any,
    output_path: str,
    schema: Optional[Dict[str, Any]] = None,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save data to JSON file with optional schema validation.

    Per CLAUDE.md P5: JSON is source of truth.
    Per CLAUDE.md Section 6 (Failure Policy): Validation before write.

    Uses atomic write (temp file + rename) to avoid partial writes.

    Args:
        data: Data to save (must be JSON serializable)
        output_path: Path to output JSON file
        schema: Optional JSON schema for validation
        indent: Indentation for pretty-printing (default 2)
        ensure_ascii: Whether to escape non-ASCII characters (default False)

    Raises:
        ValidationError: If schema validation fails
        IOError: If write fails

    Examples:
        >>> save_json({"foo": "bar"}, "/tmp/test.json")
        >>> # File created at /tmp/test.json
    """
    # Validate against schema if provided
    if schema is not None:
        try:
            validate(instance=data, schema=schema)
        except JSONSchemaValidationError as e:
            raise JSONSchemaValidationError(f"JSON validation failed: {e.message}")

    # Convert Path to string
    output_path = str(output_path)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Atomic write: write to temp file, then rename
    # This prevents partial writes if process is interrupted
    temp_fd, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_path) or ".",
        suffix=".json.tmp",
    )

    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.write("\n")  # Trailing newline

        # Atomic rename
        os.replace(temp_path, output_path)

    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise IOError(f"Failed to write JSON to {output_path}: {e}")


def load_json(
    input_path: str,
    schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
) -> Any:
    """
    Load data from JSON file with optional schema validation.

    Supports both JSON Schema (dict) and Pydantic models (BaseModel subclass).

    Args:
        input_path: Path to input JSON file
        schema: Optional JSON schema (dict) or Pydantic model class for validation

    Returns:
        Loaded data (dict, list, etc.) or Pydantic model instance if schema is a Pydantic model

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValidationError: If schema validation fails
        json.JSONDecodeError: If JSON is malformed

    Examples:
        >>> data = load_json("/tmp/test.json")
        >>> data
        {'foo': 'bar'}

        >>> # With Pydantic model
        >>> from pydantic import BaseModel
        >>> class MyModel(BaseModel):
        ...     foo: str
        >>> model = load_json("/tmp/test.json", schema=MyModel)
        >>> model.foo
        'bar'
    """
    input_path = str(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate against schema if provided
    if schema is not None:
        # Check if schema is a Pydantic model class
        if BaseModel is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
            # Use Pydantic validation
            try:
                return schema.parse_obj(data)
            except PydanticValidationError as e:
                raise ValueError(f"Pydantic validation failed for {input_path}: {e}")
        else:
            # Use JSON Schema validation (if available)
            if HAS_JSONSCHEMA:
                try:
                    validate(instance=data, schema=schema)
                except JSONSchemaValidationError as e:
                    raise ValueError(f"JSON schema validation failed for {input_path}: {e.message}")
            else:
                raise ValueError(f"JSON Schema validation requested but jsonschema module not installed. Install with: pip install jsonschema")

    return data


def json_exists(path: str) -> bool:
    """
    Check if a JSON file exists.

    Args:
        path: Path to check

    Returns:
        True if file exists and is a file (not directory)

    Examples:
        >>> json_exists("/tmp/test.json")
        True
        >>> json_exists("/nonexistent/file.json")
        False
    """
    path = str(path)
    return os.path.isfile(path)


def get_output_dir(video_name: str, base_dir: str = "videos/output") -> str:
    """
    Get output directory for a video clip.

    Per CLAUDE.md Section 3 (Output Contract):
    - Output directory: videos/output/<clip_name>/
    - Creates directory if it doesn't exist

    Args:
        video_name: Name of video clip (e.g., "clip9.mp4")
        base_dir: Base output directory (default "videos/output")

    Returns:
        Absolute path to output directory

    Examples:
        >>> get_output_dir("clip9.mp4")
        '/path/to/repo/videos/output/clip9/'
    """
    # Strip extension
    clip_name = Path(video_name).stem

    # Create output directory
    output_dir = os.path.join(base_dir, clip_name)
    os.makedirs(output_dir, exist_ok=True)

    return os.path.abspath(output_dir)


def get_artifact_path(output_dir: str, artifact_name: str) -> str:
    """
    Get path to a specific artifact within output directory.

    Per CLAUDE.md Section 3 (Output Contract):
    - Artifacts: pass1_raw.json, pass2_fragments.json, etc.

    Args:
        output_dir: Output directory path
        artifact_name: Artifact name (e.g., "pass1_raw")

    Returns:
        Absolute path to artifact JSON file

    Examples:
        >>> get_artifact_path("/path/to/output", "pass1_raw")
        '/path/to/output/pass1_raw.json'
    """
    return os.path.join(output_dir, f"{artifact_name}.json")


def list_artifacts(output_dir: str) -> Dict[str, bool]:
    """
    List all artifacts in output directory and their existence status.

    Per CLAUDE.md Section 3 (Output Contract):
    - 9 expected artifacts: pass1_raw, pass1_validation, pass2_fragments, etc.

    Args:
        output_dir: Output directory path

    Returns:
        Dict mapping artifact_name -> exists (bool)

    Examples:
        >>> list_artifacts("/path/to/output")
        {
            'pass1_raw': True,
            'pass1_validation': True,
            'pass2_fragments': False,
            'pass2b_scored_fragments': False,
            ...
        }
    """
    expected_artifacts = [
        "pass1_raw",
        "pass1_validation",
        "pass2_fragments",
        "pass2b_scored_fragments",
        "pass2_ghosts",
        "pass2_validation",
        "pass3_candidates",
        "pass3_constraints",
        "pass3_identity_commit",
        "pass3_validation",
        "ball_interpolation",
        "debug_metrics",
    ]

    status = {}
    for artifact in expected_artifacts:
        path = get_artifact_path(output_dir, artifact)
        status[artifact] = json_exists(path)

    return status


def validate_artifact(
    artifact_path: str,
    artifact_type: str,
) -> bool:
    """
    Validate an artifact against its schema.

    Args:
        artifact_path: Path to artifact JSON file
        artifact_type: Type of artifact (e.g., "pass1_raw", "pass3_identity_commit")

    Returns:
        True if validation passes

    Raises:
        ValidationError: If validation fails

    Examples:
        >>> validate_artifact("/path/to/pass1_raw.json", "pass1_raw")
        True
    """
    from ..core.schemas import get_schema

    try:
        schema = get_schema(artifact_type)
    except KeyError:
        raise ValueError(f"Unknown artifact type: {artifact_type}")

    data = load_json(artifact_path, schema=schema)

    return True  # If we got here, validation passed


def pretty_print_json(data: Any) -> str:
    """
    Convert data to pretty-printed JSON string.

    Args:
        data: Data to convert

    Returns:
        Pretty-printed JSON string

    Examples:
        >>> pretty_print_json({"foo": "bar"})
        '{\\n  "foo": "bar"\\n}'
    """
    return json.dumps(data, indent=2, ensure_ascii=False)


def compact_json(data: Any) -> str:
    """
    Convert data to compact JSON string (no whitespace).

    Args:
        data: Data to convert

    Returns:
        Compact JSON string

    Examples:
        >>> compact_json({"foo": "bar"})
        '{"foo":"bar"}'
    """
    return json.dumps(data, separators=(',', ':'), ensure_ascii=False)


def merge_json_files(input_paths: list, output_path: str) -> None:
    """
    Merge multiple JSON files into one.

    Assumes all files contain lists at top level.

    Args:
        input_paths: List of input JSON file paths
        output_path: Output JSON file path

    Examples:
        >>> merge_json_files(["/tmp/a.json", "/tmp/b.json"], "/tmp/merged.json")
    """
    merged = []

    for path in input_paths:
        data = load_json(path)
        if isinstance(data, list):
            merged.extend(data)
        else:
            raise ValueError(f"Expected list in {path}, got {type(data)}")

    save_json(merged, output_path)
