// Package tools implementa execução REAL de ferramentas de arquivo, espelhando fielmente o
// schema e os códigos de erro dos executores Python (src/tools/executors/*.py) — mesmos
// nomes de campo (`path`, `content`, `mode`, `start_line`, `end_line`), mesmos códigos de
// erro (`FILE_NOT_FOUND`, `INCOMPLETE_SOLUTION`), porque é esse o formato exato que o
// adapter foi treinado para reconhecer. Divergir os nomes de campo aqui seria o mesmo tipo
// de mismatch de formato que `D-train-prompt-mask` corrigiu no repo principal, só que do
// lado das ferramentas em vez do prompt.
//
// Todas as ferramentas operam relativas a um `workDir` (o diretório onde o `conatus` foi
// chamado) e recusam qualquer caminho que escape dele — mesma lógica de
// `SandboxContext.resolve_path` (src/security/sandbox.py): resolve o caminho absoluto e
// confere que ele continua dentro do workDir antes de tocar no disco.
package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Result é o equivalente ao ToolExecutionResult do Python — Passed vira status="ok"/"error"
// na tag <tool_result>, Data vira o corpo JSON.
type Result struct {
	Passed bool
	Data   map[string]any
}

type Executor func(args map[string]any, workDir string) Result

var Registry = map[string]Executor{
	"write_file": writeFile,
	"read_file":  readFile,
	"list_files": listFiles,
	"checker":    checker,
}

// resolvePath junta workDir + relPath, resolve pra absoluto, e confere que o resultado
// continua dentro de workDir — recusa qualquer ".." que escape, mesmo através de symlinks
// óbvios (não resolve symlink de verdade, diferente do Python `.resolve()`, mas cobre o caso
// comum de travessia por "../").
func resolvePath(workDir, relPath string) (string, bool) {
	absWorkDir, err := filepath.Abs(workDir)
	if err != nil {
		return "", false
	}
	candidate := filepath.Join(absWorkDir, relPath)
	rel, err := filepath.Rel(absWorkDir, candidate)
	if err != nil {
		return "", false
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", false
	}
	return candidate, true
}

func stringArg(args map[string]any, key, def string) string {
	if v, ok := args[key].(string); ok {
		return v
	}
	return def
}

func intArg(args map[string]any, key string, def int) int {
	// encoding/json decodifica número JSON como float64 quando o destino é map[string]any.
	if v, ok := args[key].(float64); ok {
		return int(v)
	}
	return def
}

func writeFile(args map[string]any, workDir string) Result {
	rawPath, _ := args["path"].(string)
	if rawPath == "" {
		return Result{Passed: false, Data: map[string]any{"code": "TOOL_ARGUMENT_SCHEMA_ERROR", "message": "'path' é obrigatório"}}
	}
	path, ok := resolvePath(workDir, rawPath)
	if !ok {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": fmt.Sprintf("caminho escapa do workspace: %s", rawPath)}}
	}

	mode := stringArg(args, "mode", "overwrite")
	if mode == "create" {
		if _, err := os.Stat(path); err == nil {
			return Result{Passed: false, Data: map[string]any{
				"code":    "INCOMPLETE_SOLUTION",
				"message": fmt.Sprintf("arquivo já existe e mode='create' não permite sobrescrever: %s", rawPath),
			}}
		}
	}

	content, _ := args["content"].(string)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": err.Error()}}
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": err.Error()}}
	}

	return Result{Passed: true, Data: map[string]any{
		"path":          rawPath,
		"bytes_written": len([]byte(content)),
		"mode":          mode,
	}}
}

func readFile(args map[string]any, workDir string) Result {
	rawPath, _ := args["path"].(string)
	if rawPath == "" {
		return Result{Passed: false, Data: map[string]any{"code": "TOOL_ARGUMENT_SCHEMA_ERROR", "message": "'path' é obrigatório"}}
	}
	path, ok := resolvePath(workDir, rawPath)
	if !ok {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": fmt.Sprintf("caminho escapa do workspace: %s", rawPath)}}
	}

	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": fmt.Sprintf("arquivo não encontrado no workspace: %s", rawPath)}}
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": err.Error()}}
	}

	lines := strings.SplitAfter(string(raw), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	totalLines := len(lines)

	startLine := intArg(args, "start_line", 1)
	if startLine < 1 {
		startLine = 1
	}
	endLine := intArg(args, "end_line", totalLines)
	if endLine > totalLines {
		endLine = totalLines
	}

	var selected string
	if startLine <= endLine && startLine <= totalLines {
		selected = strings.Join(lines[startLine-1:endLine], "")
	}

	return Result{Passed: true, Data: map[string]any{
		"content":     selected,
		"start_line":  startLine,
		"end_line":    endLine,
		"total_lines": totalLines,
	}}
}

func listFiles(args map[string]any, workDir string) Result {
	rawPath := stringArg(args, "path", ".")
	base, ok := resolvePath(workDir, rawPath)
	if !ok {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": fmt.Sprintf("caminho escapa do workspace: %s", rawPath)}}
	}
	info, err := os.Stat(base)
	if err != nil || !info.IsDir() {
		return Result{Passed: false, Data: map[string]any{"code": "FILE_NOT_FOUND", "message": fmt.Sprintf("diretório não encontrado no workspace: %s", rawPath)}}
	}

	maxDepth := intArg(args, "max_depth", 3)
	// Simplificação deliberada em relação ao Python (que aceita glob livre via
	// pathlib.glob): sem dependência externa de glob recursivo em Go, filtramos só por
	// extensão/substring simples via `glob`, se vier, aplicado ao nome do arquivo — não é
	// um glob completo, mas cobre o caso comum ("*.py", "*.go").
	pattern := stringArg(args, "glob", "")

	absWorkDir, _ := filepath.Abs(workDir)
	var found []string
	_ = filepath.Walk(base, func(p string, fi os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if p == base {
			return nil
		}
		rel, relErr := filepath.Rel(absWorkDir, p)
		if relErr != nil {
			return nil
		}
		depth := len(strings.Split(filepath.ToSlash(rel), "/"))
		if depth > maxDepth {
			if fi.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if pattern != "" && pattern != "**/*" {
			matched, _ := filepath.Match(pattern, fi.Name())
			if !matched {
				return nil
			}
		}
		found = append(found, filepath.ToSlash(rel))
		return nil
	})
	sort.Strings(found)

	return Result{Passed: true, Data: map[string]any{"files": found}}
}
