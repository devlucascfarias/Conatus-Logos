// Package ollamaclient fala com a API HTTP local da Ollama em modo "raw" (sem chat
// template) — mesmo contrato do OllamaModelRunner Python (src/inference/ollama_runner.py,
// D-ollama-model-runner/D-ollama-temperature-zero no PLAN.md do repo principal). A parada por
// stop-sequence é feita no lado do cliente via streaming, porque o `options.stop` nativo da
// Ollama trunca a tag de fechamento fora do texto — o harness precisa dela incluída.
package ollamaclient

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type Client struct {
	Model       string
	Host        string
	Temperature float64
	HTTPClient  *http.Client
}

func New(model, host string) *Client {
	return &Client{
		Model:       model,
		Host:        strings.TrimRight(host, "/"),
		Temperature: 0.0, // D-ollama-temperature-zero — determinismo, mesmo motivo do runner Python
		HTTPClient:  http.DefaultClient,
	}
}

type StopReason string

const (
	StopReasonSequence  StopReason = "stop_sequence"
	StopReasonMaxTokens StopReason = "max_tokens"
	StopReasonEOS       StopReason = "eos"
)

type Completion struct {
	Text        string
	StopReason  StopReason
	MatchedStop string
}

type generateRequest struct {
	Model   string  `json:"model"`
	Prompt  string  `json:"prompt"`
	Raw     bool    `json:"raw"`
	Stream  bool    `json:"stream"`
	Options options `json:"options"`
}

type options struct {
	NumPredict  int     `json:"num_predict"`
	Temperature float64 `json:"temperature"`
}

type streamChunk struct {
	Response   string `json:"response"`
	Done       bool   `json:"done"`
	DoneReason string `json:"done_reason"`
}

// Generate manda o prompt cru e chama onChunk a cada pedaço de texto recebido (para
// streaming em tempo real na UI). Retorna quando encontra uma das stop sequences (texto
// INCLUINDO a tag, mesmo contrato do TransformersModelRunner Python), quando `maxTokens` é
// atingido, ou quando o modelo emite EOS por conta própria.
func (c *Client) Generate(
	ctx context.Context, prompt string, stop []string, maxTokens int, onChunk func(delta string),
) (Completion, error) {
	body := generateRequest{
		Model:  c.Model,
		Prompt: prompt,
		Raw:    true,
		Stream: true,
		Options: options{
			NumPredict:  maxTokens,
			Temperature: c.Temperature,
		},
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return Completion{}, fmt.Errorf("codificando payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Host+"/api/generate", strings.NewReader(string(payload)))
	if err != nil {
		return Completion{}, fmt.Errorf("montando requisição: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return Completion{}, fmt.Errorf("chamando Ollama em %s: %w", c.Host, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Completion{}, fmt.Errorf("Ollama retornou status %d — o servidor está rodando (`ollama serve`) e o modelo existe (`ollama list`)?", resp.StatusCode)
	}

	var text strings.Builder
	var matchedStop string
	var doneReason string

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var chunk streamChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			return Completion{}, fmt.Errorf("decodificando resposta da Ollama: %w", err)
		}
		if chunk.Response != "" {
			text.WriteString(chunk.Response)
			onChunk(chunk.Response)
		}

		current := text.String()
		for _, candidate := range stop {
			if candidate != "" && strings.HasSuffix(current, candidate) {
				matchedStop = candidate
				break
			}
		}
		if matchedStop != "" {
			break
		}
		if chunk.Done {
			doneReason = chunk.DoneReason
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return Completion{}, fmt.Errorf("lendo stream da Ollama: %w", err)
	}

	if matchedStop != "" {
		return Completion{Text: text.String(), StopReason: StopReasonSequence, MatchedStop: matchedStop}, nil
	}
	if doneReason == "length" {
		return Completion{Text: text.String(), StopReason: StopReasonMaxTokens}, nil
	}
	return Completion{Text: text.String(), StopReason: StopReasonEOS}, nil
}
