package ollamaclient

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newFakeOllamaServer sobe um servidor HTTP real (httptest) que responde /api/generate com
// as chunks NDJSON dadas, uma por linha — mesmo formato de streaming da Ollama de verdade.
func newFakeOllamaServer(t *testing.T, chunks []map[string]any, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decodificando corpo da requisição: %v", err)
			}
			*captured = body
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		enc := json.NewEncoder(w)
		for _, chunk := range chunks {
			if err := enc.Encode(chunk); err != nil {
				t.Fatalf("codificando chunk fake: %v", err)
			}
		}
	}))
}

func TestGenerateUsesRawModeAndSendsPromptUnmodified(t *testing.T) {
	var captured map[string]any
	srv := newFakeOllamaServer(t, []map[string]any{
		{"response": "ok", "done": true, "done_reason": "stop"},
	}, &captured)
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	_, err := client.Generate(context.Background(), "PROMPT CRU EXATO", nil, 10, func(string) {})
	if err != nil {
		t.Fatalf("erro inesperado: %v", err)
	}

	if captured["raw"] != true {
		t.Errorf("esperava raw=true, veio %v", captured["raw"])
	}
	if captured["prompt"] != "PROMPT CRU EXATO" {
		t.Errorf("prompt foi modificado: %v", captured["prompt"])
	}
	if captured["model"] != "logos-v2" {
		t.Errorf("model incorreto: %v", captured["model"])
	}
	options, _ := captured["options"].(map[string]any)
	if options["temperature"] != 0.0 {
		t.Errorf("esperava temperature=0.0 por padrão, veio %v", options["temperature"])
	}
}

func TestGenerateStopsWithTagIncludedInText(t *testing.T) {
	srv := newFakeOllamaServer(t, []map[string]any{
		{"response": "<think>oi</think>", "done": false},
		{"response": "<tool_call", "done": false},
		{"response": ">", "done": false},
		{"response": "{}", "done": false},
		{"response": "</tool_call>", "done": false},
		{"response": "texto que nao deveria ser lido", "done": true, "done_reason": "stop"},
	}, nil)
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	var received strings.Builder
	completion, err := client.Generate(
		context.Background(), "prompt", []string{"</tool_call>", "</final>"}, 100,
		func(delta string) { received.WriteString(delta) },
	)
	if err != nil {
		t.Fatalf("erro inesperado: %v", err)
	}

	if completion.StopReason != StopReasonSequence {
		t.Errorf("esperava stop_sequence, veio %v", completion.StopReason)
	}
	if completion.MatchedStop != "</tool_call>" {
		t.Errorf("matched stop incorreto: %q", completion.MatchedStop)
	}
	if !strings.HasSuffix(completion.Text, "</tool_call>") {
		t.Errorf("texto deveria terminar com a tag de parada, veio: %q", completion.Text)
	}
	if strings.Contains(completion.Text, "texto que nao deveria ser lido") {
		t.Errorf("texto após a parada não deveria ter sido incluído: %q", completion.Text)
	}
	// onChunk também não deveria ter recebido o texto pós-parada.
	if strings.Contains(received.String(), "texto que nao deveria ser lido") {
		t.Errorf("onChunk recebeu texto pós-parada: %q", received.String())
	}
}

func TestGenerateReturnsMaxTokensWhenDoneReasonIsLength(t *testing.T) {
	srv := newFakeOllamaServer(t, []map[string]any{
		{"response": "texto sem tag de parada", "done": false},
		{"response": "", "done": true, "done_reason": "length"},
	}, nil)
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	completion, err := client.Generate(context.Background(), "prompt", []string{"</final>"}, 5, func(string) {})
	if err != nil {
		t.Fatalf("erro inesperado: %v", err)
	}
	if completion.StopReason != StopReasonMaxTokens {
		t.Errorf("esperava max_tokens, veio %v", completion.StopReason)
	}
}

func TestGenerateReturnsEOSWhenModelStopsWithoutMatchingOurStop(t *testing.T) {
	srv := newFakeOllamaServer(t, []map[string]any{
		{"response": "resposta curta", "done": true, "done_reason": "stop"},
	}, nil)
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	completion, err := client.Generate(context.Background(), "prompt", []string{"</final>"}, 100, func(string) {})
	if err != nil {
		t.Fatalf("erro inesperado: %v", err)
	}
	if completion.StopReason != StopReasonEOS {
		t.Errorf("esperava eos, veio %v", completion.StopReason)
	}
	if completion.Text != "resposta curta" {
		t.Errorf("texto inesperado: %q", completion.Text)
	}
}

func TestGenerateReturnsErrorOnNonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	_, err := client.Generate(context.Background(), "prompt", nil, 10, func(string) {})
	if err == nil {
		t.Fatal("esperava erro para status 500, veio nil")
	}
}

func TestGenerateCapturesPromptEvalCountEvenWhenStoppingEarly(t *testing.T) {
	// prompt_eval_count chega logo no início (calculado no prefill, antes do primeiro
	// token de saída) — precisa continuar disponível mesmo quando cortamos a leitura por
	// stop-sequence antes da Ollama sinalizar done=true por conta própria, que é o caso
	// comum (o harness quase sempre para antes do fim natural da geração).
	srv := newFakeOllamaServer(t, []map[string]any{
		{"response": "<final>oi", "done": false, "prompt_eval_count": 123},
		{"response": "</final>", "done": false},
		{"response": "texto que nunca deveria ser lido", "done": true, "done_reason": "stop"},
	}, nil)
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	completion, err := client.Generate(context.Background(), "prompt", []string{"</final>"}, 100, func(string) {})
	if err != nil {
		t.Fatalf("erro inesperado: %v", err)
	}
	if completion.PromptEvalCount != 123 {
		t.Errorf("esperava PromptEvalCount=123, veio %d", completion.PromptEvalCount)
	}
	if completion.StopReason != StopReasonSequence {
		t.Errorf("esperava stop_sequence, veio %v", completion.StopReason)
	}
}

func TestGenerateSendsExplicitNumCtx(t *testing.T) {
	var captured map[string]any
	srv := newFakeOllamaServer(t, []map[string]any{
		{"response": "ok", "done": true, "done_reason": "stop"},
	}, &captured)
	defer srv.Close()

	client := New("logos-v2", srv.URL)
	client.NumCtx = 8192
	_, err := client.Generate(context.Background(), "prompt", nil, 10, func(string) {})
	if err != nil {
		t.Fatalf("erro inesperado: %v", err)
	}

	options, _ := captured["options"].(map[string]any)
	if options["num_ctx"] != float64(8192) {
		t.Errorf("esperava num_ctx=8192, veio %v", options["num_ctx"])
	}
}
