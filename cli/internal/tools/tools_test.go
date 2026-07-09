package tools

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteFileCreatesRealFile(t *testing.T) {
	dir := t.TempDir()
	result := writeFile(map[string]any{"path": "sub/hello.py", "content": "print('oi')\n"}, dir)

	if !result.Passed {
		t.Fatalf("esperava sucesso, veio: %+v", result.Data)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "sub", "hello.py"))
	if err != nil {
		t.Fatalf("arquivo não foi criado: %v", err)
	}
	if string(raw) != "print('oi')\n" {
		t.Errorf("conteúdo incorreto: %q", raw)
	}
	if result.Data["bytes_written"] != len("print('oi')\n") {
		t.Errorf("bytes_written incorreto: %v", result.Data["bytes_written"])
	}
}

func TestWriteFileRejectsPathEscapingWorkDir(t *testing.T) {
	dir := t.TempDir()
	result := writeFile(map[string]any{"path": "../escapou.txt", "content": "x"}, dir)

	if result.Passed {
		t.Fatal("esperava falha por travessia de caminho, veio sucesso")
	}
	if result.Data["code"] != "FILE_NOT_FOUND" {
		t.Errorf("código de erro incorreto: %v", result.Data["code"])
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "escapou.txt")); err == nil {
		t.Error("arquivo foi criado fora do workDir")
	}
}

func TestWriteFileModeCreateRefusesToOverwrite(t *testing.T) {
	dir := t.TempDir()
	writeFile(map[string]any{"path": "x.txt", "content": "original"}, dir)

	result := writeFile(map[string]any{"path": "x.txt", "content": "novo", "mode": "create"}, dir)
	if result.Passed {
		t.Fatal("esperava falha (mode=create com arquivo já existente), veio sucesso")
	}
	if result.Data["code"] != "INCOMPLETE_SOLUTION" {
		t.Errorf("código de erro incorreto: %v", result.Data["code"])
	}

	raw, _ := os.ReadFile(filepath.Join(dir, "x.txt"))
	if string(raw) != "original" {
		t.Error("conteúdo original foi sobrescrito apesar do mode=create ter sido rejeitado")
	}
}

func TestReadFileReturnsRealContent(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("linha1\nlinha2\nlinha3\n"), 0o644)

	result := readFile(map[string]any{"path": "a.txt"}, dir)
	if !result.Passed {
		t.Fatalf("esperava sucesso, veio: %+v", result.Data)
	}
	if result.Data["content"] != "linha1\nlinha2\nlinha3\n" {
		t.Errorf("conteúdo incorreto: %q", result.Data["content"])
	}
	if result.Data["total_lines"] != 3 {
		t.Errorf("total_lines incorreto: %v", result.Data["total_lines"])
	}
}

func TestReadFileRespectsLineRange(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("linha1\nlinha2\nlinha3\n"), 0o644)

	result := readFile(map[string]any{"path": "a.txt", "start_line": float64(2), "end_line": float64(2)}, dir)
	if !result.Passed {
		t.Fatalf("esperava sucesso, veio: %+v", result.Data)
	}
	if result.Data["content"] != "linha2\n" {
		t.Errorf("conteúdo incorreto pro intervalo pedido: %q", result.Data["content"])
	}
}

func TestReadFileReturnsFileNotFoundForMissingFile(t *testing.T) {
	dir := t.TempDir()
	result := readFile(map[string]any{"path": "nao-existe.txt"}, dir)

	if result.Passed {
		t.Fatal("esperava falha, veio sucesso")
	}
	if result.Data["code"] != "FILE_NOT_FOUND" {
		t.Errorf("código de erro incorreto: %v", result.Data["code"])
	}
}

func TestListFilesFindsRealFilesRecursively(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "sub"), 0o755)
	os.WriteFile(filepath.Join(dir, "a.py"), []byte("x"), 0o644)
	os.WriteFile(filepath.Join(dir, "sub", "b.py"), []byte("x"), 0o644)

	result := listFiles(map[string]any{}, dir)
	if !result.Passed {
		t.Fatalf("esperava sucesso, veio: %+v", result.Data)
	}
	files, _ := result.Data["files"].([]string)
	if len(files) < 2 {
		t.Fatalf("esperava pelo menos 2 arquivos, veio %v", files)
	}
}

func TestListFilesRejectsPathEscapingWorkDir(t *testing.T) {
	dir := t.TempDir()
	result := listFiles(map[string]any{"path": "../"}, dir)

	if result.Passed {
		t.Fatal("esperava falha por travessia de caminho, veio sucesso")
	}
	if result.Data["code"] != "FILE_NOT_FOUND" {
		t.Errorf("código de erro incorreto: %v", result.Data["code"])
	}
}

func TestResolvePathAllowsNestedPathsInsideWorkDir(t *testing.T) {
	dir := t.TempDir()
	_, ok := resolvePath(dir, "a/b/c.txt")
	if !ok {
		t.Error("caminho aninhado legítimo dentro do workDir foi rejeitado")
	}
}

func TestResolvePathRejectsTraversalOutsideWorkDir(t *testing.T) {
	dir := t.TempDir()
	_, ok := resolvePath(dir, "../../etc/passwd")
	if ok {
		t.Error("travessia de caminho deveria ter sido rejeitada")
	}
}
