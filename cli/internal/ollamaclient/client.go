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
	// NumCtx é o tamanho da janela de contexto pedido explicitamente à Ollama
	// (options.num_ctx) — sem fixar isso, a Ollama usa um default próprio não documentado
	// de forma confiável no retorno da API, e o contador de contexto da UI não teria um
	// denominador conhecido pra mostrar "usado/limite".
	NumCtx     int
	HTTPClient *http.Client
}

func New(model, host string) *Client {
	return &Client{
		Model:       model,
		Host:        strings.TrimRight(host, "/"),
		Temperature: 0.0, // D-ollama-temperature-zero — determinismo, mesmo motivo do runner Python
		NumCtx:      4096,
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
	// PromptEvalCount é o número REAL de tokens do prompt de entrada, reportado pela
	// própria Ollama (campo prompt_eval_count) — não é uma estimativa nossa. Ao contrário
	// de EvalCount (tokens gerados), esse valor fica disponível cedo no streaming
	// (calculado no prefill, antes do primeiro token de saída), então continua confiável
	// mesmo quando paramos a leitura antes da Ollama sinalizar done=true por conta própria
	// (nosso corte por stop-sequence quase sempre acontece antes disso).
	PromptEvalCount int
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
	NumCtx      int     `json:"num_ctx"`
}

type streamChunk struct {
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason"`
	PromptEvalCount int    `json:"prompt_eval_count"`
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
			NumCtx:      c.NumCtx,
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
	var promptEvalCount int

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
		if chunk.PromptEvalCount > 0 {
			promptEvalCount = chunk.PromptEvalCount
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
		return Completion{Text: text.String(), StopReason: StopReasonSequence, MatchedStop: matchedStop, PromptEvalCount: promptEvalCount}, nil
	}
	if doneReason == "length" {
		return Completion{Text: text.String(), StopReason: StopReasonMaxTokens, PromptEvalCount: promptEvalCount}, nil
	}
	return Completion{Text: text.String(), StopReason: StopReasonEOS, PromptEvalCount: promptEvalCount}, nil
}

// CountTokens devolve o número REAL de tokens que `prompt` ocupa, sem gerar nenhum texto
// (`num_predict: 0` — só prefill, sem decode). Existe porque `prompt_eval_count` só é
// confiável quando a Ollama chega no seu PRÓPRIO done=true — e o harness quase sempre corta
// a leitura antes disso (stop-sequence customizada), então capturar esse campo durante uma
// geração normal (Generate) não funciona na prática. Um `num_predict=0` sempre completa
// rápido (sem decodificar token nenhum) e sempre chega no done=true de verdade, então o
// número volta confiável sem sacrificar a parada antecipada das chamadas de geração normais.
func (c *Client) CountTokens(ctx context.Context, prompt string) (int, error) {
	completion, err := c.Generate(ctx, prompt, nil, 0, func(string) {})
	if err != nil {
		return 0, err
	}
	return completion.PromptEvalCount, nil
}
