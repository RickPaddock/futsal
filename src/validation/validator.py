"""
Main validation orchestrator.

Per CLAUDE.md Section 6 (Failure Policy):
- Validation runs BEFORE writing JSON
- Failed pass writes nothing (not even partial artifacts)
- Fail-fast: execution halts on any error violation
- Warnings are logged but don't block execution
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from ..core.data_models import (
    Pass1Output,
    Pass2COutput,
    Pass3COutput,
    BallInterpolationOutput,
    ValidationViolation,
    ValidationResult,
    ScoredFragment,
)


class Validator:
    """
    Main validation orchestrator.

    Per CLAUDE.md P6: Fail-fast on validation errors.

    Usage:
        validator = Validator()
        result = validator.validate_pass1(pass1_output)
        if not result.passed:
            raise ValidationError(result)
    """

    def __init__(self, fail_fast: bool = True):
        """
        Initialize validator.

        Args:
            fail_fast: If True, stop validation on first error (default True)
        """
        self.fail_fast = fail_fast

    def validate_pass1(self, pass1_output: Pass1Output) -> ValidationResult:
        """
        Validate Pass 1 output.

        Args:
            pass1_output: Pass 1 output data

        Returns:
            ValidationResult with violations (if any)
        """
        from .pass1_rules import validate_pass1

        violations = validate_pass1(pass1_output)

        return self._build_result("pass1", violations)

    def validate_pass2(self, pass2c_output: Pass2COutput) -> ValidationResult:
        """
        Validate Pass 2 output (2A/2B/2C combined).

        Args:
            pass2c_output: Pass 2C output data

        Returns:
            ValidationResult with violations (if any)
        """
        from .pass2_rules import validate_pass2

        violations = validate_pass2(pass2c_output)

        return self._build_result("pass2", violations)

    def validate_pass3(
        self,
        pass3c_output: Pass3COutput,
        fragments: List[ScoredFragment],
    ) -> ValidationResult:
        """
        Validate Pass 3 output (3A/3B/3C combined).

        Args:
            pass3c_output: Pass 3C output data
            fragments: List of fragments (for R2/R3 validation)

        Returns:
            ValidationResult with violations (if any)
        """
        from .pass3_rules import validate_pass3

        violations = validate_pass3(pass3c_output, fragments)

        return self._build_result("pass3", violations)

    def validate_ball_interpolation(
        self,
        ball_output: BallInterpolationOutput,
        total_frames: int,
    ) -> ValidationResult:
        """
        Validate ball interpolation output.

        Args:
            ball_output: Ball interpolation output
            total_frames: Total frames in video

        Returns:
            ValidationResult with violations (if any)
        """
        from .ball_rules import (
            validate_ball_interpolation,
            validate_ball_state_consistency,
        )

        violations = []
        violations.extend(validate_ball_interpolation(ball_output, total_frames))
        violations.extend(validate_ball_state_consistency(ball_output.ball_positions))

        return self._build_result("ball", violations)

    def _build_result(
        self,
        pass_name: str,
        violations: List[ValidationViolation],
    ) -> ValidationResult:
        """
        Build ValidationResult from violations.

        Args:
            pass_name: Name of pass being validated
            violations: List of violations

        Returns:
            ValidationResult
        """
        # Separate errors and warnings
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]

        # Validation passes if no errors
        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            violations=errors,
            warnings=warnings,
            timestamp=datetime.utcnow().isoformat() + "Z",
            pass_name=pass_name,
        )

    def validate_all(
        self,
        pass1_output: Optional[Pass1Output] = None,
        pass2c_output: Optional[Pass2COutput] = None,
        pass3c_output: Optional[Pass3COutput] = None,
        fragments: Optional[List[ScoredFragment]] = None,
        ball_output: Optional[BallInterpolationOutput] = None,
        total_frames: Optional[int] = None,
    ) -> Dict[str, ValidationResult]:
        """
        Run all validations.

        Args:
            pass1_output: Pass 1 output (optional)
            pass2c_output: Pass 2C output (optional)
            pass3c_output: Pass 3C output (optional)
            fragments: Fragments for Pass 3 validation (optional)
            ball_output: Ball interpolation output (optional)
            total_frames: Total frames in video (optional)

        Returns:
            Dict mapping pass_name -> ValidationResult
        """
        results = {}

        if pass1_output:
            results["pass1"] = self.validate_pass1(pass1_output)

        if pass2c_output:
            results["pass2"] = self.validate_pass2(pass2c_output)

        if pass3c_output and fragments:
            results["pass3"] = self.validate_pass3(pass3c_output, fragments)

        if ball_output and total_frames:
            results["ball"] = self.validate_ball_interpolation(ball_output, total_frames)

        return results

    def check_all_passed(self, results: Dict[str, ValidationResult]) -> bool:
        """
        Check if all validation results passed.

        Args:
            results: Dict mapping pass_name -> ValidationResult

        Returns:
            True if all passed, False otherwise
        """
        return all(result.passed for result in results.values())


class ValidationError(Exception):
    """
    Exception raised when validation fails.

    Per CLAUDE.md P6: Fail-fast on validation errors.
    """

    def __init__(self, result: ValidationResult):
        """
        Initialize validation error.

        Args:
            result: ValidationResult with violations
        """
        self.result = result

        # Build error message
        error_count = len(result.violations)
        warning_count = len(result.warnings)

        message = f"Validation failed for {result.pass_name}: {error_count} errors, {warning_count} warnings\n"

        # Add first few errors
        for i, violation in enumerate(result.violations[:5]):
            message += f"  [{violation.rule}] {violation.message}\n"

        if len(result.violations) > 5:
            message += f"  ... and {len(result.violations) - 5} more errors\n"

        super().__init__(message)


def validate_and_raise(validator: Validator, pass_name: str, **kwargs) -> ValidationResult:
    """
    Convenience function to validate and raise if failed.

    Per CLAUDE.md P6: Fail-fast policy.

    Args:
        validator: Validator instance
        pass_name: Pass name ("pass1", "pass2", "pass3", "ball")
        **kwargs: Arguments for validation

    Returns:
        ValidationResult (only if passed)

    Raises:
        ValidationError: If validation failed

    Examples:
        >>> validator = Validator()
        >>> result = validate_and_raise(validator, "pass1", pass1_output=data)
        >>> # If validation failed, exception is raised before this line
    """
    if pass_name == "pass1":
        result = validator.validate_pass1(kwargs["pass1_output"])
    elif pass_name == "pass2":
        result = validator.validate_pass2(kwargs["pass2c_output"])
    elif pass_name == "pass3":
        result = validator.validate_pass3(
            kwargs["pass3c_output"],
            kwargs["fragments"],
        )
    elif pass_name == "ball":
        result = validator.validate_ball_interpolation(
            kwargs["ball_output"],
            kwargs["total_frames"],
        )
    else:
        raise ValueError(f"Unknown pass_name: {pass_name}")

    if not result.passed:
        raise ValidationError(result)

    return result
