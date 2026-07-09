package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
	go Run(context.Background(), client, "olá", nil, events)
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
	go Run(context.Background(), client, "segundo pedido", history, events)
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
	go Run(context.Background(), client, "olá", nil, events)
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
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		enc := json.NewEncoder(w)
		if calls == 1 {
			_ = enc.Encode(map[string]any{
				"response": `<tool_call name="write_file">{}</tool_call>`, "done": true, "done_reason": "stop",
			})
			return
		}
		_ = enc.Encode(map[string]any{"response": "<final>ok</final>", "done": true, "done_reason": "stop"})
	}))
	defer srv.Close()

	client := ollamaclient.New("logos-v2", srv.URL)
	events := make(chan Event)
	go Run(context.Background(), client, "crie um arquivo", nil, events)
	got := drain(events)

	var sawUnsupported bool
	for _, ev := range got {
		if strings.Contains(ev.Delta, "UNSUPPORTED_TOOL") && strings.Contains(ev.Delta, "write_file") {
			sawUnsupported = true
		}
	}
	if !sawUnsupported {
		t.Errorf("esperava um tool_result UNSUPPORTED_TOOL pra write_file, eventos: %+v", got)
	}
	if calls != 2 {
		t.Errorf("esperava 2 chamadas ao servidor (tool_call + retomada), veio %d", calls)
	}
}
