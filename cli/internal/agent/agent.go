// Package agent é uma porta MÍNIMA do loop do harness real (src/harness/loop.py,
// run_agent_loop) — o suficiente para orquestrar geração + parada por stop-sequence contra o
// Ollama local. NÃO executa ferramentas de verdade ainda (write_file/checker/shell) — quando
// o modelo emite um <tool_call>, o loop injeta o mesmo erro que o harness real usaria para uma
// ferramenta desconhecida/desabilitada (UNSUPPORTED_TOOL), porque nenhuma ferramenta está
// registrada nesta build. Isso é uma resposta real do "harness" (decidida por código, não
// pelo modelo) — não é fabricação disfarçada.
//
// System prompt e stop-sequences copiados verbatim de src/harness/system_prompt.py e
// AgentLoopConfig (loop.py) — divergir esse texto do treino reproduziria o mesmo tipo de
// mismatch de formato que D-train-prompt-mask corrigiu no repo principal.
//
// Histórico de sessão (Turn/Run com history): ATENÇÃO — o dataset de treino do harness é
// inteiramente de turno único (um pedido, uma trajetória, sem exemplo de conversa
// encadeada). Concatenar turnos anteriores no mesmo formato [USER]/[ASSISTANT] é a extensão
// mais fiel possível ao padrão treinado (reaproveita a mesma gramática, não inventa uma
// nova), mas não há garantia de que o modelo generalize pra isso tão bem quanto generaliza
// pro turno único — é território não coberto pelo treino.
package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/devlucascfarias/Conatus-Logos/cli/internal/ollamaclient"
	"github.com/devlucascfarias/Conatus-Logos/cli/internal/segments"
)

const SystemPrompt = "Você é Praxis, um agente de engenharia de software. Raciocine em <think>, use " +
	"<tool_call name=\"...\"> quando precisar de uma ferramenta, e responda ao usuário só " +
	"dentro de <final>."

const (
	maxSteps         = 8
	maxTokensPerStep = 512
)

var stopSequences = []string{"</tool_call>", "</final>"}

// Turn é um turno já concluído da sessão — UserRequest é o pedido, RawText é a trajetória
// COMPLETA gerada pro turno (com <think>/<tool_call>/<tool_result>/<final>, não só a
// resposta final), porque é esse o formato que o [ASSISTANT] sempre teve durante o treino.
type Turn struct {
	UserRequest string
	RawText     string
}

// Event é emitido incrementalmente durante a execução do loop — Delta é texto novo pra
// renderizar ao vivo na UI; Done marca o fim (sucesso ou erro, ver Err). ContextTokens é o
// número REAL de tokens do prompt (system + histórico + turno atual) reportado pela própria
// Ollama após a chamada mais recente — 0 se ainda não disponível.
type Event struct {
	Delta         string
	Done          bool
	Err           error
	ContextTokens int
}

// Run executa o loop e envia eventos em events até fechar o canal. Deve ser chamado numa
// goroutine própria — bloqueia até terminar. `history` é opcional (nil/vazio pra sessão sem
// contexto anterior).
func Run(ctx context.Context, client *ollamaclient.Client, userRequest string, history []Turn, events chan<- Event) {
	defer close(events)

	var historyPrefix strings.Builder
	for _, turn := range history {
		historyPrefix.WriteString("\n\n[USER]\n")
		historyPrefix.WriteString(turn.UserRequest)
		historyPrefix.WriteString("\n\n[ASSISTANT]\n")
		historyPrefix.WriteString(turn.RawText)
	}

	rawText := ""

	for step := 0; step < maxSteps; step++ {
		prompt := SystemPrompt + historyPrefix.String() + "\n\n[USER]\n" + userRequest + "\n\n[ASSISTANT]\n" + rawText

		completion, err := client.Generate(ctx, prompt, stopSequences, maxTokensPerStep, func(delta string) {
			events <- Event{Delta: delta}
		})
		if err != nil {
			events <- Event{Err: err, Done: true}
			return
		}
		rawText += completion.Text
		if completion.PromptEvalCount > 0 {
			events <- Event{ContextTokens: completion.PromptEvalCount}
		}

		switch completion.MatchedStop {
		case "</final>":
			events <- Event{Done: true}
			return
		case "</tool_call>":
			toolName := lastToolCallName(completion.Text)
			toolResult := fmt.Sprintf(
				`<tool_result name="%s" status="error">{"code": "UNSUPPORTED_TOOL", "message": "ferramenta desconhecida/desabilitada: %s (execução de ferramentas ainda não implementada nesta build da CLI)"}</tool_result>`,
				toolName, toolName,
			)
			rawText += toolResult
			events <- Event{Delta: toolResult}
			continue
		default:
			// max_tokens ou eos sem tag reconhecida — não dá pra continuar com segurança
			// (mesmo espírito de MAX_STEPS_EXCEEDED do harness real), encerra aqui.
			events <- Event{Done: true}
			return
		}
	}

	events <- Event{Done: true}
}

func lastToolCallName(text string) string {
	segs := segments.Parse(text)
	name := "unknown"
	for _, seg := range segs {
		if seg.Kind == segments.KindToolCall {
			name = seg.ToolName
		}
	}
	return name
}
