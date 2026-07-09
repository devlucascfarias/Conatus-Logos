package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/devlucascfarias/Conatus-Logos/cli/internal/ollamaclient"
)

func newFakeServer(t *testing.T, capturedPrompt *string, response string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decodificando corpo: %v", err)
		}
		if capturedPrompt != nil {
			*capturedPrompt = body["prompt"].(string)
		}
		enc := json.NewEncoder(w)
		_ = enc.Encode(map[string]any{"response": response, "done": true, "done_reason": "stop", "prompt_eval_count": 42})
	}))
}

func drain(events <-chan Event) []Event {
	var out []Event
	for ev := range events {
		out = append(out, ev)
	}
	return out
}

func TestRunSendsNoHistoryPrefixOnFirstTurn(t *testing.T) {
	var capturedPrompt string
	srv := newFakeServer(t, &capturedPrompt, "<final>oi</final>")
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "olá", nil, t.TempDir(), events)
	drain(events)

	wantPrefix := SystemPrompt + "\n\n[USER]\nolá\n\n[ASSISTANT]\n"
	if capturedPrompt != wantPrefix {
		t.Errorf("prompt do primeiro turno incorreto:\nveio: %q\nesperado: %q", capturedPrompt, wantPrefix)
	}
}

func TestRunIncludesFullRawTextOfPastTurnsInPrompt(t *testing.T) {
	var capturedPrompt string
	srv := newFakeServer(t, &capturedPrompt, "<final>ok</final>")
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	history := []Turn{
		{UserRequest: "primeiro pedido", RawText: "<think>pensei</think><final>primeira resposta</final>"},
	}
	events := make(chan Event)
	go Run(context.Background(), client, "segundo pedido", history, t.TempDir(), events)
	drain(events)

	if !strings.Contains(capturedPrompt, "primeiro pedido") {
		t.Errorf("pedido do turno anterior não apareceu no prompt: %q", capturedPrompt)
	}
	if !strings.Contains(capturedPrompt, "<think>pensei</think><final>primeira resposta</final>") {
		t.Errorf("trajetória completa do turno anterior não apareceu no prompt: %q", capturedPrompt)
	}
	if !strings.HasSuffix(capturedPrompt, "[USER]\nsegundo pedido\n\n[ASSISTANT]\n") {
		t.Errorf("turno atual não veio depois do histórico: %q", capturedPrompt)
	}
}

func TestRunEmitsContextTokensFromRealPromptEvalCount(t *testing.T) {
	srv := newFakeServer(t, nil, "<final>oi</final>")
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "olá", nil, t.TempDir(), events)
	got := drain(events)

	found := false
	for _, ev := range got {
		if ev.ContextTokens == 42 {
			found = true
		}
	}
	if !found {
		t.Errorf("esperava um evento com ContextTokens=42 (prompt_eval_count real), eventos: %+v", got)
	}
}

func TestRunUnsupportedToolInjectsRealHarnessErrorFormat(t *testing.T) {
	// "shell" ainda não está registrado em tools.Registry — path certo pra exercitar o
	// fallback UNSUPPORTED_TOOL (diferente de write_file/read_file/list_files/checker, que
	// agora executam de verdade).
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		enc := json.NewEncoder(w)
		if calls == 1 {
			_ = enc.Encode(map[string]any{
				"response": `<tool_call name="shell">{}</tool_call>`, "done": true, "done_reason": "stop",
			})
			return
		}
		_ = enc.Encode(map[string]any{"response": "<final>ok</final>", "done": true, "done_reason": "stop"})
	}))
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "rode um comando", nil, t.TempDir(), events)
	got := drain(events)

	var sawUnsupported bool
	for _, ev := range got {
		if strings.Contains(ev.Delta, "UNSUPPORTED_TOOL") && strings.Contains(ev.Delta, "shell") {
			sawUnsupported = true
		}
	}
	if !sawUnsupported {
		t.Errorf("esperava um tool_result UNSUPPORTED_TOOL pra shell, eventos: %+v", got)
	}
	if calls != 2 {
		t.Errorf("esperava 2 chamadas ao servidor (tool_call + retomada), veio %d", calls)
	}
}

func TestRunWriteFileActuallyCreatesFileInWorkDir(t *testing.T) {
	workDir := t.TempDir()
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		enc := json.NewEncoder(w)
		if calls == 1 {
			_ = enc.Encode(map[string]any{
				"response": `<tool_call name="write_file">{"path": "hello.py", "content": "print('oi')\n"}</tool_call>`,
				"done":     true, "done_reason": "stop",
			})
			return
		}
		_ = enc.Encode(map[string]any{"response": "<final>criado</final>", "done": true, "done_reason": "stop"})
	}))
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "crie hello.py", nil, workDir, events)
	got := drain(events)

	raw, err := os.ReadFile(filepath.Join(workDir, "hello.py"))
	if err != nil {
		t.Fatalf("arquivo não foi criado de verdade no workDir: %v", err)
	}
	if string(raw) != "print('oi')\n" {
		t.Errorf("conteúdo do arquivo incorreto: %q", string(raw))
	}

	var sawOK bool
	for _, ev := range got {
		if strings.Contains(ev.Delta, `status="ok"`) && strings.Contains(ev.Delta, "bytes_written") {
			sawOK = true
		}
	}
	if !sawOK {
		t.Errorf("esperava um tool_result de sucesso pra write_file, eventos: %+v", got)
	}
}

func TestRunWriteFileRejectsPathEscapingWorkDir(t *testing.T) {
	workDir := t.TempDir()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		enc := json.NewEncoder(w)
		_ = enc.Encode(map[string]any{
			"response": `<tool_call name="write_file">{"path": "../fora.txt", "content": "x"}</tool_call>`,
			"done":     true, "done_reason": "stop",
		})
	}))
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "crie um arquivo fora", nil, workDir, events)
	got := drain(events)

	var sawEscape bool
	for _, ev := range got {
		if strings.Contains(ev.Delta, "FILE_NOT_FOUND") && strings.Contains(ev.Delta, "escapa do workspace") {
			sawEscape = true
		}
	}
	if !sawEscape {
		t.Errorf("esperava rejeição por caminho escapando do workDir, eventos: %+v", got)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(workDir), "fora.txt")); err == nil {
		t.Error("arquivo foi criado fora do workDir — travessia de caminho não foi bloqueada")
	}
}

func TestRunMalformedToolCallJSONReturnsRealParseError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		enc := json.NewEncoder(w)
		_ = enc.Encode(map[string]any{
			"response": `<tool_call name="write_file">{"path": "x.py", "content": "sem fechar as aspas</tool_call>`,
			"done":     true, "done_reason": "stop",
		})
	}))
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "crie x.py", nil, t.TempDir(), events)
	got := drain(events)

	var sawParseError bool
	for _, ev := range got {
		if strings.Contains(ev.Delta, "TOOL_CALL_PARSE_ERROR") {
			sawParseError = true
		}
	}
	if !sawParseError {
		t.Errorf("esperava TOOL_CALL_PARSE_ERROR pra JSON malformado (quebra de linha crua), eventos: %+v", got)
	}
}

func TestRunCheckerActuallyValidatesRealPythonSyntax(t *testing.T) {
	if _, err := exec.LookPath("python"); err != nil {
		t.Skip("python não disponível neste ambiente de teste")
	}

	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		enc := json.NewEncoder(w)
		if calls == 1 {
			_ = enc.Encode(map[string]any{
				"response": `<tool_call name="checker">{"language": "python", "operation": "syntax_check", "files": [{"path": "a.py", "content": "def f(:\n"}]}</tool_call>`,
				"done":     true, "done_reason": "stop",
			})
			return
		}
		_ = enc.Encode(map[string]any{"response": "<final>tem erro</final>", "done": true, "done_reason": "stop"})
	}))
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "valide esse código", nil, t.TempDir(), events)
	got := drain(events)

	var sawSyntaxError bool
	for _, ev := range got {
		if strings.Contains(ev.Delta, `"checker"`) && strings.Contains(ev.Delta, "SYNTAX_ERROR") {
			sawSyntaxError = true
		}
	}
	if !sawSyntaxError {
		t.Errorf("esperava um SYNTAX_ERROR real (py_compile de verdade) pro código inválido, eventos: %+v", got)
	}
}
