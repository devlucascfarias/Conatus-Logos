// checker.go espelha `src/checker/backends/python_backend.py` — mesmos códigos de erro
// (`SYNTAX_ERROR`, `TEST_FAILURE`, `MISSING_DEPENDENCY`, `TIMEOUT`, `RUNTIME_ERROR`,
// `UNSUPPORTED_LANGUAGE`, `INCOMPLETE_SOLUTION`), mesmas operações (`syntax_check`/`compile`,
// `compile_and_test`, `run`, `lint`). Só o backend Python está portado — o dataset de
// treino/testes desta sessão usa quase exclusivamente `language="python"`; Go fica pra uma
// futura leva, não é fingido aqui (linguagem sem backend cai em `UNSUPPORTED_LANGUAGE` real,
// igual o Python faz pra qualquer linguagem sem backend registrado).
//
// Diferente de write_file/read_file/list_files, o checker NUNCA toca o workDir real — os
// arquivos recebidos em `files` são materializados num diretório TEMPORÁRIO isolado (mesmo
// padrão do `tempfile.TemporaryDirectory` do Python), pra não sujar o diretório de trabalho
// do usuário com artefatos de validação.
package tools

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

var tracebackLocationRe = regexp.MustCompile(`File "([^"]+)", line (\d+)`)

func checker(args map[string]any, _ string) Result {
	language := stringArg(args, "language", "")
	operation := stringArg(args, "operation", "")
	timeoutMs := intArg(args, "timeout_ms", 15000)
	entrypoint := stringArg(args, "entrypoint", "")

	if language != "python" {
		return checkResult(false, nil, "", "", map[string]any{"language": language, "duration_ms": 0},
			"UNSUPPORTED_LANGUAGE", fmt.Sprintf("nenhum backend de checker registrado para '%s'", language))
	}

	rawFiles, _ := args["files"].([]any)
	if len(rawFiles) == 0 {
		return checkResult(false, nil, "", "", map[string]any{"language": language, "duration_ms": 0},
			"TOOL_ARGUMENT_SCHEMA_ERROR", "'files' é obrigatório e não pode ser vazio")
	}

	tmpDir, err := os.MkdirTemp("", "conatus_checker_py_")
	if err != nil {
		return checkResult(false, nil, "", "", map[string]any{"language": language, "duration_ms": 0},
			"SANDBOX_ERROR", err.Error())
	}
	defer os.RemoveAll(tmpDir)

	var pyFiles []string
	for _, rf := range rawFiles {
		f, _ := rf.(map[string]any)
		path, _ := f["path"].(string)
		content, _ := f["content"].(string)
		full := filepath.Join(tmpDir, path)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			continue
		}
		os.WriteFile(full, []byte(content), 0o644)
		if strings.HasSuffix(path, ".py") {
			pyFiles = append(pyFiles, path)
		}
	}

	switch operation {
	case "syntax_check", "compile":
		return syntaxCheck(tmpDir, pyFiles, timeoutMs)
	case "compile_and_test":
		result := syntaxCheck(tmpDir, pyFiles, timeoutMs)
		if !result.Passed {
			return result
		}
		return runPytest(tmpDir, timeoutMs)
	case "run":
		return runEntrypoint(tmpDir, entrypoint, timeoutMs)
	case "lint":
		return runLint(tmpDir, timeoutMs)
	default:
		return checkResult(false, nil, "", "", map[string]any{"language": language, "duration_ms": 0},
			"INVALID_OUTPUT_FORMAT", fmt.Sprintf("operação desconhecida: %s", operation))
	}
}

func pythonExecutable() (string, error) {
	for _, candidate := range []string{"python", "python3", "py"} {
		if path, err := exec.LookPath(candidate); err == nil {
			return path, nil
		}
	}
	return "", errors.New("nenhum executável Python encontrado no PATH (tentado: python, python3, py)")
}

func runCommand(ctx context.Context, dir string, timeoutMs int, name string, args ...string) (stdout, stderr string, timedOut bool, durationMs int) {
	ctxTimeout, cancel := context.WithTimeout(ctx, time.Duration(timeoutMs)*time.Millisecond)
	defer cancel()

	cmd := exec.CommandContext(ctxTimeout, name, args...)
	cmd.Dir = dir

	var outBuf, errBuf strings.Builder
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf

	start := time.Now()
	err := cmd.Run()
	durationMs = int(time.Since(start).Milliseconds())

	if ctxTimeout.Err() == context.DeadlineExceeded {
		return outBuf.String(), errBuf.String(), true, durationMs
	}
	_ = err
	return outBuf.String(), errBuf.String(), false, durationMs
}

func syntaxCheck(dir string, pyFiles []string, timeoutMs int) Result {
	python, err := pythonExecutable()
	if err != nil {
		return checkResult(false, nil, "", "", map[string]any{"language": "python", "duration_ms": 0}, "MISSING_DEPENDENCY", err.Error())
	}
	if len(pyFiles) == 0 {
		return checkResult(false, nil, "", "", map[string]any{"language": "python", "duration_ms": 0},
			"UNSUPPORTED_LANGUAGE", "nenhum arquivo .py em 'files'")
	}

	var checkErrors []map[string]any
	var stdoutParts, stderrParts []string
	totalMs := 0

	for _, f := range pyFiles {
		stdout, stderr, timedOut, ms := runCommand(context.Background(), dir, timeoutMs, python, "-m", "py_compile", f)
		totalMs += ms
		stdoutParts = append(stdoutParts, stdout)
		stderrParts = append(stderrParts, stderr)
		if timedOut {
			checkErrors = append(checkErrors, map[string]any{"code": "TIMEOUT", "message": fmt.Sprintf("timeout ao compilar %s", f), "file": f})
			continue
		}
		if stderr != "" {
			file, line := extractLocation(stderr, dir)
			if file == "" {
				file = f
			}
			entry := map[string]any{"code": "SYNTAX_ERROR", "message": strings.TrimSpace(stderr), "file": file}
			if line > 0 {
				entry["line"] = line
			}
			checkErrors = append(checkErrors, entry)
		}
	}

	passed := len(checkErrors) == 0
	return checkResult(passed, checkErrors, strings.Join(stdoutParts, "\n"), strings.Join(stderrParts, "\n"),
		map[string]any{"language": "python", "duration_ms": totalMs}, "", "")
}

func runPytest(dir string, timeoutMs int) Result {
	python, err := pythonExecutable()
	if err != nil {
		return checkResult(false, nil, "", "", map[string]any{"language": "python", "duration_ms": 0}, "MISSING_DEPENDENCY", err.Error())
	}

	stdout, stderr, timedOut, ms := runCommand(context.Background(), dir, timeoutMs, python, "-m", "pytest", "-q", "--no-header")
	metadata := map[string]any{"language": "python", "duration_ms": ms}
	if timedOut {
		return checkResult(false, []map[string]any{{"code": "TIMEOUT", "message": "timeout ao rodar pytest"}}, stdout, stderr, metadata, "", "")
	}

	combined := stdout + stderr
	passed := !strings.Contains(combined, "FAILED") && !strings.Contains(combined, "ERROR") && !strings.Contains(combined, "Error")
	if passed {
		return checkResult(true, nil, stdout, stderr, metadata, "", "")
	}

	code := "RUNTIME_ERROR"
	switch {
	case strings.Contains(combined, "ModuleNotFoundError") || strings.Contains(combined, "No module named"):
		code = "MISSING_DEPENDENCY"
	case strings.Contains(combined, "FAILED") || strings.Contains(combined, "assert"):
		code = "TEST_FAILURE"
	}
	msg := combined
	if len(msg) > 2000 {
		msg = msg[len(msg)-2000:]
	}
	return checkResult(false, []map[string]any{{"code": code, "message": strings.TrimSpace(msg)}}, stdout, stderr, metadata, "", "")
}

func runEntrypoint(dir, entrypoint string, timeoutMs int) Result {
	if entrypoint == "" {
		return checkResult(false, nil, "", "", map[string]any{"language": "python", "duration_ms": 0},
			"INCOMPLETE_SOLUTION", "operation=run requer 'entrypoint'")
	}
	python, err := pythonExecutable()
	if err != nil {
		return checkResult(false, nil, "", "", map[string]any{"language": "python", "duration_ms": 0}, "MISSING_DEPENDENCY", err.Error())
	}

	stdout, stderr, timedOut, ms := runCommand(context.Background(), dir, timeoutMs, python, entrypoint)
	metadata := map[string]any{"language": "python", "duration_ms": ms}
	if timedOut {
		return checkResult(false, []map[string]any{{"code": "TIMEOUT", "message": fmt.Sprintf("timeout ao executar %s", entrypoint)}}, stdout, stderr, metadata, "", "")
	}
	if stderr == "" {
		return checkResult(true, nil, stdout, stderr, metadata, "", "")
	}

	file, line := extractLocation(stderr, dir)
	if file == "" {
		file = entrypoint
	}
	msg := strings.TrimSpace(stderr)
	if len(msg) > 2000 {
		msg = msg[len(msg)-2000:]
	}
	entry := map[string]any{"code": "RUNTIME_ERROR", "message": msg, "file": file}
	if line > 0 {
		entry["line"] = line
	}
	return checkResult(false, []map[string]any{entry}, stdout, stderr, metadata, "", "")
}

func runLint(dir string, timeoutMs int) Result {
	if _, err := exec.LookPath("ruff"); err != nil {
		return checkResult(false, nil, "", "", map[string]any{"language": "python", "duration_ms": 0},
			"MISSING_DEPENDENCY", "ruff não disponível no ambiente")
	}
	stdout, stderr, _, ms := runCommand(context.Background(), dir, timeoutMs, "ruff", "check", ".")
	metadata := map[string]any{"language": "python", "duration_ms": ms}
	if strings.TrimSpace(stdout) == "" && strings.TrimSpace(stderr) == "" {
		return checkResult(true, nil, stdout, stderr, metadata, "", "")
	}
	msg := stdout
	if len(msg) > 2000 {
		msg = msg[len(msg)-2000:]
	}
	return checkResult(false, []map[string]any{{"code": "SYNTAX_ERROR", "message": strings.TrimSpace(msg)}}, stdout, stderr, metadata, "", "")
}

func extractLocation(stderr, baseDir string) (file string, line int) {
	matches := tracebackLocationRe.FindAllStringSubmatch(stderr, -1)
	if len(matches) == 0 {
		return "", 0
	}
	last := matches[len(matches)-1]
	rawPath := last[1]
	if rel, err := filepath.Rel(baseDir, rawPath); err == nil && !strings.HasPrefix(rel, "..") {
		file = filepath.ToSlash(rel)
	} else {
		file = rawPath
	}
	fmt.Sscanf(last[2], "%d", &line)
	return file, line
}

// checkResult monta o Result no mesmo formato de CheckResult.to_json() do Python
// (passed/errors/stdout/stderr/metadata) — se code/message vierem preenchidos, cria um
// único erro nessa forma (atalho pros casos de erro estrutural antes mesmo de tentar rodar
// algo, como UNSUPPORTED_LANGUAGE ou TOOL_ARGUMENT_SCHEMA_ERROR).
func checkResult(passed bool, checkErrors []map[string]any, stdout, stderr string, metadata map[string]any, code, message string) Result {
	if code != "" {
		checkErrors = []map[string]any{{"code": code, "message": message}}
	}
	if checkErrors == nil {
		checkErrors = []map[string]any{}
	}
	return Result{
		Passed: passed,
		Data: map[string]any{
			"passed":   passed,
			"errors":   checkErrors,
			"stdout":   stdout,
			"stderr":   stderr,
			"metadata": metadata,
		},
	}
}
