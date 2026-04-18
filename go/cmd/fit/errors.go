package main

import (
	"errors"
	"fmt"
	"syscall"

	"hop.top/kit/cli"
)

// Error codes for structured error correction.
const (
	errCodeDatasetMissing = "DATASET_MISSING"
	errCodeDatasetLoad    = "DATASET_LOAD_FAILED"
	errCodeDatasetParse   = "DATASET_PARSE_FAILED"
	errCodeDatasetFormat  = "DATASET_FORMAT_UNSUPPORTED"
	errCodeTraceRead      = "TRACE_READ_FAILED"
	errCodeTraceParse     = "TRACE_PARSE_FAILED"
	errCodePortInUse      = "PORT_IN_USE"
)

func errDatasetMissing() *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodeDatasetMissing,
		Message: "--dataset flag is required",
		Cause:   "no dataset path provided",
		Fix:     "pass --dataset path/to/dataset.json",
		Alternatives: []string{
			"fit eval --dataset examples/eval-dataset.json",
		},
		Retryable: false,
	}
}

func errDatasetLoad(path string, cause error) *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodeDatasetLoad,
		Message: fmt.Sprintf("cannot load dataset: %s", path),
		Cause:   cause.Error(),
		Fix:     fmt.Sprintf("verify %s exists and is readable", path),
		Alternatives: []string{
			"check file permissions",
			"use an absolute path",
		},
		Retryable: true,
	}
}

func errDatasetParse(path string, cause error) *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodeDatasetParse,
		Message: fmt.Sprintf("cannot parse dataset: %s", path),
		Cause:   cause.Error(),
		Fix: fmt.Sprintf(
			"ensure %s contains valid JSON", path,
		),
		Retryable: false,
	}
}

func errDatasetFormat(ext string) *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodeDatasetFormat,
		Message: fmt.Sprintf("unsupported dataset format: %s", ext),
		Cause:   fmt.Sprintf("extension %s is not supported", ext),
		Fix:     "use .json format for dataset files",
		Retryable: false,
	}
}

func errTraceRead(path string, cause error) *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodeTraceRead,
		Message: fmt.Sprintf("cannot read trace: %s", path),
		Cause:   cause.Error(),
		Fix:     "verify session ID and step number are correct",
		Alternatives: []string{
			"fit trace list — to see available sessions",
			"fit trace list --dir <path> — if using a custom traces directory",
		},
		Retryable: true,
	}
}

func errTraceParse(path string, cause error) *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodeTraceParse,
		Message: fmt.Sprintf("cannot parse trace: %s", path),
		Cause:   cause.Error(),
		Fix:     "ensure the trace file is valid YAML conforming to the xrr trace schema",
		Retryable: false,
	}
}

// isAddrInUse returns true if err indicates the address is already bound.
func isAddrInUse(err error) bool {
	var errno syscall.Errno
	if errors.As(err, &errno) {
		return errno == syscall.EADDRINUSE
	}
	return false
}

func errPortInUse(addr string, cause error) *cli.CorrectedError {
	return &cli.CorrectedError{
		Code:    errCodePortInUse,
		Message: fmt.Sprintf("cannot listen on %s", addr),
		Cause:   cause.Error(),
		Fix:     "use --addr :N with a different port or kill the existing process",
		Alternatives: []string{
			"lsof -i :PORT — to find the process using the port",
			"FIT_ADDR=:9090 fit serve — to use a different port via env",
		},
		Retryable: true,
	}
}
