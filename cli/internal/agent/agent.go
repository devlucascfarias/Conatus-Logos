// Package agent é uma porta MÍNIMA do loop do harness real (src/harness/loop.py,
// run_agent_loop) — o suficiente para orquestrar geração + parada por stop-sequence contra o
// Ollama local, com execução REAL de ferramentas de arquivo (internal/tools — write_file,
// read_file, list_files). checker/shell/web_search ainda não estão portados: um <tool_call>
// pra eles recebe o mesmo erro que o harness real usaria para uma ferramenta desconhecida/
// desabilitada (UNSUPPORTED_TOOL). Isso é uma resposta real do "harness" (decidida por
// código, não pelo modelo) — não é fabricação disfarçada.
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
	"encoding/json"
	"fmt"
	"strings"

	"github.com/devlucascfarias/Conatus-Logos/cli/internal/ollamaclient"
	"github.com/devlucascfarias/Conatus-Logos/cli/internal/segments"
	"github.com/devlucascfarias/Conatus-Logos/cli/internal/tools"
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
// renderizar ao vivo na UI; Done marca o fim (sucesso ou erro, ver Err). TotalTokens é o
// número REAL de tokens do prompt completo do turno (system + histórico + este turno
// inteiro), medido com uma chamada dedicada de contagem (ver ollamaclient.CountTokens) no
// momento em que o turno termina — 0 se ainda não disponível (turno ainda em andamento, ou a
// medição falhou; nunca é uma estimativa).
type Event struct {
	Delta       string
	Done        bool
	Err         error
	TotalTokens int
}

// Run executa o loop e envia eventos em events até fechar o canal. Deve ser chamado numa
// goroutine própria — bloqueia até terminar. `history` é opcional (nil/vazio pra sessão sem
// contexto anterior). `workDir` é o diretório onde as ferramentas de arquivo operam de
// verdade — normalmente o diretório em que o `conatus` foi chamado.
func Run(ctx context.Context, client *ollamaclient.Client, userRequest string, history []Turn, workDir string, events chan<- Event) {
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

		switch completion.MatchedStop {
		case "</final>":
			finishTurn(ctx, client, prompt, completion.Text, events)
			return
		case "</tool_call>":
			toolResult := executeToolCall(completion.Text, workDir)
			rawText += toolResult
			events <- Event{Delta: toolResult}
			continue
		default:
			// max_tokens ou eos sem tag reconhecida — não dá pra continuar com segurança
			// (mesmo espírito de MAX_STEPS_EXCEEDED do harness real), encerra aqui.
			finishTurn(ctx, client, prompt, completion.Text, events)
			return
		}
	}

	events <- Event{Done: true}
}

// finishTurn mede o tamanho real do prompt final do turno (prompt usado na última chamada +
// o texto que ela gerou = exatamente o que entraria como [ASSISTANT] de um próximo turno) via
// uma chamada dedicada e barata (num_predict=0, só prefill) antes de sinalizar Done — dá um
// número real pro contador de contexto da UI sem depender de capturar prompt_eval_count no
// meio de uma geração normal (não confiável, ver CountTokens).
func finishTurn(ctx context.Context, client *ollamaclient.Client, promptSoFar, lastText string, events chan<- Event) {
	finalPrompt := promptSoFar + lastText
	if total, err := client.CountTokens(ctx, finalPrompt); err == nil {
		events <- Event{TotalTokens: total}
	}
	events <- Event{Done: true}
}

// executeToolCall roda a ferramenta de verdade se ela existir em tools.Registry; senão,
// devolve o mesmo UNSUPPORTED_TOOL que o harness Python devolveria (loop.py,
// _handle_tool_call). Se o JSON do tool_call vier malformado, devolve o mesmo
// TOOL_CALL_PARSE_ERROR que o harness real produziria via src.parsers.parse_last_segment —
// nunca fabrica um <tool_result> de sucesso.
func executeToolCall(generatedText, workDir string) string {
	name, argsJSON := lastToolCall(generatedText)

	executor, registered := tools.Registry[name]
	if !registered {
		return fmt.Sprintf(
			`<tool_result name="%s" status="error">{"code": "UNSUPPORTED_TOOL", "message": "ferramenta desconhecida/desabilitada: %s"}</tool_result>`,
			name, name,
		)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
		msg, _ := json.Marshal(err.Error())
		return fmt.Sprintf(
			`<tool_result name="%s" status="error">{"code": "TOOL_CALL_PARSE_ERROR", "message": %s}</tool_result>`,
			name, msg,
		)
	}

	result := executor(args, workDir)
	body, err := json.Marshal(result.Data)
	if err != nil {
		return fmt.Sprintf(
			`<tool_result name="%s" status="error">{"code": "TOOL_CALL_PARSE_ERROR", "message": "falha ao serializar resultado"}</tool_result>`,
			name,
		)
	}

	status := "error"
	if result.Passed {
		status = "ok"
	}
	return fmt.Sprintf(`<tool_result name="%s" status="%s">%s</tool_result>`, name, status, body)
}

func lastToolCall(text string) (name, argsJSON string) {
	segs := segments.Parse(text)
	name = "unknown"
	for _, seg := range segs {
		if seg.Kind == segments.KindToolCall {
			name = seg.ToolName
			argsJSON = seg.Body
		}
	}
	return name, argsJSON
}
