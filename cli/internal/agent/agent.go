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
package agent

import (
	"context"
	"fmt"

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

// Event é emitido incrementalmente durante a execução do loop — Delta é texto novo pra
// renderizar ao vivo na UI; Done marca o fim (sucesso ou erro, ver Err).
type Event struct {
	Delta string
	Done  bool
	Err   error
}

// Run executa o loop e envia eventos em events até fechar o canal. Deve ser chamado numa
// goroutine própria — bloqueia até terminar.
func Run(ctx context.Context, client *ollamaclient.Client, userRequest string, events chan<- Event) {
	defer close(events)

	rawText := ""

	for step := 0; step < maxSteps; step++ {
		prompt := SystemPrompt + "\n\n[USER]\n" + userRequest + "\n\n[ASSISTANT]\n" + rawText

		completion, err := client.Generate(ctx, prompt, stopSequences, maxTokensPerStep, func(delta string) {
			events <- Event{Delta: delta}
		})
		if err != nil {
			events <- Event{Err: err, Done: true}
			return
		}
		rawText += completion.Text

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
