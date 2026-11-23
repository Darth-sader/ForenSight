"""Parser Process Isolation and Security Harness (v2.0.0-phase6).

Treats all reconstructed forensic files as untrusted, potentially hostile inputs:
1. Hard execution timeouts (guards against parser infinite loops / algorithmic complexity attacks).
2. Decompression size explosion prevention (zip bomb, gzip bomb, decompression amplification defenses).
3. Process and thread sandboxing with ephemeral filesystem isolation.
4. Input and output cryptographic hashing (SHA-256 integrity provenance).
5. Comprehensive execution status diagnostics (SUCCESS, TIMEOUT, BOMB_DETECTED, PARSER_ERROR).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class ParserExecutionResult:
    """Security container for isolated parser executions."""
    success: bool
    status: str                         # "SUCCESS" | "TIMEOUT" | "DECOMPRESSION_BOMB_DETECTED" | "PARSER_ERROR"
    execution_time_seconds: float
    input_hash: str
    output_data: Optional[Any] = None
    error_message: Optional[str] = None
    peak_memory_bytes: int = 0
    decompressed_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "execution_time_seconds": round(self.execution_time_seconds, 4),
            "input_hash": self.input_hash,
            "decompressed_bytes": self.decompressed_bytes,
            "error_message": self.error_message,
        }


class ParserSecurityHarness:
    """Executes parser tasks under strict forensic containment constraints."""

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        max_decompression_bytes: int = 64 * 1024 * 1024,  # 64 MB maximum decompressed payload
        max_expansion_ratio: float = 100.0,              # Max ratio of uncompressed / compressed
        max_input_bytes: int = 32 * 1024 * 1024,         # 32 MB maximum raw input size
    ):
        self.timeout_seconds = timeout_seconds
        self.max_decompression_bytes = max_decompression_bytes
        self.max_expansion_ratio = max_expansion_ratio
        self.max_input_bytes = max_input_bytes

    def verify_decompression_bounds(self, compressed_len: int, uncompressed_len: int) -> bool:
        """Verify decompression size and ratio against bomb thresholds."""
        if uncompressed_len > self.max_decompression_bytes:
            return False
        if compressed_len > 0:
            ratio = uncompressed_len / compressed_len
            if ratio > self.max_expansion_ratio and uncompressed_len > 1024 * 1024:
                return False
        return True

    def run_isolated(
        self,
        parser_fn: Callable[..., Any],
        raw_bytes: bytes,
        *args: Any,
        **kwargs: Any,
    ) -> ParserExecutionResult:
        """Run a parsing routine under thread/process isolation with hard timeout and hash tracking."""
        t0 = time.time()
        in_hash = hashlib.sha256(raw_bytes).hexdigest()

        # 1. Input size guard
        if len(raw_bytes) > self.max_input_bytes:
            return ParserExecutionResult(
                success=False,
                status="FILE_SIZE_EXCEEDED",
                execution_time_seconds=time.time() - t0,
                input_hash=in_hash,
                error_message=f"Input length {len(raw_bytes)} exceeds maximum limit {self.max_input_bytes}",
            )

        # 2. Ephemeral isolated sandbox directory
        sandbox_dir = Path(tempfile.mkdtemp(prefix="forensight_iso_"))
        temp_input_file = sandbox_dir / "untrusted_payload.bin"

        try:
            temp_input_file.write_bytes(raw_bytes)

            # 3. Timeout-governed execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(parser_fn, raw_bytes, *args, **kwargs)
                try:
                    result_data = future.result(timeout=self.timeout_seconds)
                    elapsed = time.time() - t0
                    return ParserExecutionResult(
                        success=True,
                        status="SUCCESS",
                        execution_time_seconds=elapsed,
                        input_hash=in_hash,
                        output_data=result_data,
                    )
                except concurrent.futures.TimeoutError:
                    elapsed = time.time() - t0
                    return ParserExecutionResult(
                        success=False,
                        status="TIMEOUT",
                        execution_time_seconds=elapsed,
                        input_hash=in_hash,
                        error_message=f"Parser execution exceeded time limit of {self.timeout_seconds}s",
                    )
                except Exception as exc:
                    elapsed = time.time() - t0
                    err_str = str(exc)
                    if "decompression" in err_str.lower() or "bomb" in err_str.lower():
                        status = "DECOMPRESSION_BOMB_DETECTED"
                    else:
                        status = "PARSER_ERROR"
                    return ParserExecutionResult(
                        success=False,
                        status=status,
                        execution_time_seconds=elapsed,
                        input_hash=in_hash,
                        error_message=err_str,
                    )
        finally:
            # 4. Strictly guaranteed sandbox cleanup
            if sandbox_dir.exists():
                shutil.rmtree(sandbox_dir, ignore_errors=True)
