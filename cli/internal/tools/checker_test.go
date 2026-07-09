package tools

import (
	"os"
	"testing"
)

func requirePython(t *testing.T) {
	t.Helper()
	if _, err := pythonExecutable(); err != nil {
		t.Skip("python não disponível neste ambiente de teste — pulando (mesmo espírito de MISSING_DEPENDENCY real)")
	}
}

func TestCheckerSyntaxCheckPassesForValidPython(t *testing.T) {
	requirePython(t)
	result := checker(map[string]any{
		"language":  "python",
		"operation": "syntax_check",
		"files":     []any{map[string]any{"path": "a.py", "content": "print('oi')\n"}},
	}, "")

	if !result.Passed {
		t.Fatalf("esperava sucesso, veio: %+v", result.Data)
	}
	if result.Data["passed"] != true {
		t.Errorf("campo passed incorreto: %v", result.Data["passed"])
	}
}

func TestCheckerSyntaxCheckFailsForRealSyntaxError(t *testing.T) {
	requirePython(t)
	result := checker(map[string]any{
		"language":  "python",
		"operation": "syntax_check",
		"files":     []any{map[string]any{"path": "a.py", "content": "def f(:\n    pass\n"}},
	}, "")

	if result.Passed {
		t.Fatal("esperava falha real de sintaxe, veio sucesso")
	}
	errs, _ := result.Data["errors"].([]map[string]any)
	if len(errs) == 0 || errs[0]["code"] != "SYNTAX_ERROR" {
		t.Errorf("esperava erro SYNTAX_ERROR, veio: %+v", result.Data["errors"])
	}
}

func TestCheckerCompileAndTestRunsRealPytest(t *testing.T) {
	requirePython(t)
	result := checker(map[string]any{
		"language":  "python",
		"operation": "compile_and_test",
		"files": []any{
			map[string]any{"path": "add.py", "content": "def add(a, b):\n    return a + b\n"},
			map[string]any{"path": "test_add.py", "content": "from add import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"},
		},
	}, "")

	if !result.Passed {
		t.Fatalf("esperava sucesso, veio: %+v", result.Data)
	}
}

func TestCheckerCompileAndTestFailsOnRealAssertionFailure(t *testing.T) {
	requirePython(t)
	result := checker(map[string]any{
		"language":  "python",
		"operation": "compile_and_test",
		"files": []any{
			map[string]any{"path": "add.py", "content": "def add(a, b):\n    return a - b\n"},
			map[string]any{"path": "test_add.py", "content": "from add import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"},
		},
	}, "")

	if result.Passed {
		t.Fatal("esperava falha real de teste (add subtrai em vez de somar), veio sucesso")
	}
}

func TestCheckerReturnsUnsupportedLanguageForNonPython(t *testing.T) {
	result := checker(map[string]any{
		"language":  "javascript",
		"operation": "syntax_check",
		"files":     []any{map[string]any{"path": "a.js", "content": "console.log(1)"}},
	}, "")

	if result.Passed {
		t.Fatal("esperava falha (linguagem não suportada), veio sucesso")
	}
	errs, _ := result.Data["errors"].([]map[string]any)
	if len(errs) == 0 || errs[0]["code"] != "UNSUPPORTED_LANGUAGE" {
		t.Errorf("esperava UNSUPPORTED_LANGUAGE, veio: %+v", result.Data["errors"])
	}
}

func TestCheckerDoesNotTouchRealWorkDir(t *testing.T) {
	requirePython(t)
	workDir := t.TempDir()
	checker(map[string]any{
		"language":  "python",
		"operation": "syntax_check",
		"files":     []any{map[string]any{"path": "a.py", "content": "print(1)\n"}},
	}, workDir)

	entries, err := os.ReadDir(workDir)
	if err != nil {
		t.Fatalf("erro lendo workDir: %v", err)
	}
	if len(entries) != 0 {
		t.Errorf("checker escreveu no workDir real (deveria usar um diretório temporário isolado): %v", entries)
	}
}
