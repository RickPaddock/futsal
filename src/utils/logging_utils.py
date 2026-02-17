"""
Structured logging utilities.

Per CLAUDE.md Section 7 (Utilities):
- Structured JSON logging for debugging
- Pass-specific loggers
- Audit trail for all decisions
- Everything auditable without watching video
"""

import logging
import sys
import json
from typing import Any, Dict, Optional
from datetime import datetime
from pathlib import Path


class StructuredLogger:
    """
    Structured logger for pipeline execution.

    Per CLAUDE.md Section 6 (Everything Auditable):
    - Logs decisions, splits, constraints, etc.
    - Outputs both human-readable and machine-readable formats
    - Pass-specific logging with context
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (e.g., "pass1", "pass2a", "pass3c")
            level: Logging level (default INFO)
            log_file: Optional log file path
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            fmt='[%(asctime)s] %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (JSON structured)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = StructuredFormatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs) -> None:
        """
        Set context fields that will be included in all log messages.

        Args:
            **kwargs: Context fields (e.g., video_name="clip9", pass_name="pass1")

        Examples:
            >>> logger.set_context(video_name="clip9.mp4", pass_name="pass1")
            >>> logger.info("Processing started")
            # Log includes: {"video_name": "clip9.mp4", "pass_name": "pass1", "message": "Processing started"}
        """
        self.context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context fields."""
        self.context.clear()

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured data."""
        self.logger.info(message, extra={"context": self.context, "data": kwargs})

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional structured data."""
        self.logger.warning(message, extra={"context": self.context, "data": kwargs})

    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional structured data."""
        self.logger.error(message, extra={"context": self.context, "data": kwargs})

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional structured data."""
        self.logger.debug(message, extra={"context": self.context, "data": kwargs})

    def log_split(
        self,
        fragment_id: str,
        split_frame: int,
        reason: str,
        **kwargs
    ) -> None:
        """
        Log a fragment split decision.

        Per CLAUDE.md Section 5 (Pass 2A): All splits are auditable.

        Args:
            fragment_id: Fragment being split
            split_frame: Frame where split occurs
            reason: Reason for split
            **kwargs: Additional split metadata
        """
        self.info(
            f"SPLIT: {fragment_id} at frame {split_frame}",
            fragment_id=fragment_id,
            split_frame=split_frame,
            reason=reason,
            **kwargs,
        )

    def log_constraint(
        self,
        constraint_type: str,
        fragment_ids: list,
        reason: str,
        **kwargs
    ) -> None:
        """
        Log a constraint creation.

        Per CLAUDE.md Section 5 (Pass 3B): All constraints are auditable.

        Args:
            constraint_type: Type of constraint (MUST_SAME, CANNOT_SAME, SOFT_SAME)
            fragment_ids: Fragments involved
            reason: Reason for constraint
            **kwargs: Additional constraint metadata
        """
        self.info(
            f"CONSTRAINT: {constraint_type} on {fragment_ids}",
            constraint_type=constraint_type,
            fragment_ids=fragment_ids,
            reason=reason,
            **kwargs,
        )

    def log_identity_decision(
        self,
        fragment_id: str,
        player_id: str,
        team: str,
        jersey: int,
        method: str,
        confidence: float,
        **kwargs
    ) -> None:
        """
        Log an identity assignment.

        Per CLAUDE.md P3: Identity is decided at Pass 3C lock point.

        Args:
            fragment_id: Fragment ID
            player_id: Assigned player ID
            team: Assigned team
            jersey: Assigned jersey number
            method: Assignment method
            confidence: Assignment confidence
            **kwargs: Additional decision metadata
        """
        self.info(
            f"IDENTITY: {fragment_id} → {player_id}",
            fragment_id=fragment_id,
            player_id=player_id,
            team=team,
            jersey=jersey,
            method=method,
            confidence=confidence,
            **kwargs,
        )

    def log_ghost_creation(
        self,
        fragment_id: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
        reason: str,
        **kwargs
    ) -> None:
        """
        Log a ghost fragment creation.

        Per CLAUDE.md R4: Players never disappear.

        Args:
            fragment_id: Ghost fragment ID
            track_id: Original track ID
            start_frame: Ghost start frame
            end_frame: Ghost end frame
            reason: Reason for ghost creation
            **kwargs: Additional ghost metadata
        """
        self.info(
            f"GHOST: {fragment_id} for track {track_id} [{start_frame}-{end_frame}]",
            fragment_id=fragment_id,
            track_id=track_id,
            start_frame=start_frame,
            end_frame=end_frame,
            reason=reason,
            is_ghost=True,
            **kwargs,
        )

    def log_validation_result(
        self,
        passed: bool,
        violations: int,
        warnings: int,
        **kwargs
    ) -> None:
        """
        Log validation result.

        Per CLAUDE.md Section 6 (Failure Policy): Fail-fast on validation errors.

        Args:
            passed: Whether validation passed
            violations: Number of violations (errors)
            warnings: Number of warnings
            **kwargs: Additional validation metadata
        """
        level = self.error if not passed else self.info
        level(
            f"VALIDATION: {'PASSED' if passed else 'FAILED'} ({violations} errors, {warnings} warnings)",
            validation_passed=passed,
            violations=violations,
            warnings=warnings,
            **kwargs,
        )


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON lines for machine parsing.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON string
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context if present
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add structured data if present
        if hasattr(record, "data"):
            log_entry["data"] = record.data

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(
    pass_name: str,
    output_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> StructuredLogger:
    """
    Get a structured logger for a specific pass.

    Args:
        pass_name: Pass name (e.g., "pass1", "pass2a", "pass3c")
        output_dir: Optional output directory for log file
        level: Logging level (default INFO)

    Returns:
        StructuredLogger instance

    Examples:
        >>> logger = get_logger("pass1", output_dir="/path/to/output")
        >>> logger.set_context(video_name="clip9.mp4")
        >>> logger.info("Processing started")
    """
    log_file = None
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_file = str(Path(output_dir) / f"{pass_name}.log")

    logger = StructuredLogger(pass_name, level=level, log_file=log_file)
    logger.set_context(pass_name=pass_name)

    return logger


def log_pass_start(logger: StructuredLogger, pass_name: str, **kwargs) -> None:
    """
    Log the start of a pass.

    Args:
        logger: Logger instance
        pass_name: Pass name
        **kwargs: Additional context
    """
    logger.info(
        f"=== {pass_name.upper()} START ===",
        event="pass_start",
        **kwargs,
    )


def log_pass_end(
    logger: StructuredLogger,
    pass_name: str,
    duration_seconds: float,
    **kwargs
) -> None:
    """
    Log the end of a pass.

    Args:
        logger: Logger instance
        pass_name: Pass name
        duration_seconds: Pass duration in seconds
        **kwargs: Additional context (e.g., fragments_created, ghosts_created)
    """
    logger.info(
        f"=== {pass_name.upper()} END ({duration_seconds:.2f}s) ===",
        event="pass_end",
        duration_seconds=duration_seconds,
        **kwargs,
    )


def log_pipeline_summary(
    logger: StructuredLogger,
    total_duration: float,
    artifacts_created: list,
    validation_passed: bool,
    **kwargs
) -> None:
    """
    Log pipeline summary.

    Args:
        logger: Logger instance
        total_duration: Total pipeline duration in seconds
        artifacts_created: List of artifact names created
        validation_passed: Whether all validations passed
        **kwargs: Additional summary data
    """
    logger.info(
        f"PIPELINE COMPLETE ({total_duration:.2f}s)",
        event="pipeline_complete",
        total_duration_seconds=total_duration,
        artifacts_created=artifacts_created,
        validation_passed=validation_passed,
        **kwargs,
    )
