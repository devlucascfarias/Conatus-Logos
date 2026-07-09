// Package segments faz um parsing simples (só para exibição na TUI) da gramática de
// trajetória do harness Python (src/parsers no repo principal): <think>, <tool_call
// name="...">, <tool_result name="..." status="...">, <final>. Não valida schema nem detecta
// fabricação — isso é responsabilidade do harness real (Python por enquanto, Go depois, seção
// M8 do PLAN.md). Este pacote só transforma raw_text em blocos que a UI sabe estilizar.
package segments

import (
	"regexp"
)

type Kind int

const (
	KindThink Kind = iota
	KindToolCall
	KindToolResult
	KindFinal
	KindUnknown
)

type Segment struct {
	Kind     Kind
	ToolName string
	Status   string // só preenchido para KindToolResult ("ok" | "error")
	Body     string
}

var (
	thinkRe      = regexp.MustCompile(`(?s)<think>(.*?)</think>`)
	toolCallRe   = regexp.MustCompile(`(?s)<tool_call name="([^"]*)">(.*?)</tool_call>`)
	toolResultRe = regexp.MustCompile(`(?s)<tool_result name="([^"]*)" status="([^"]*)">(.*?)</tool_result>`)
	finalRe      = regexp.MustCompile(`(?s)<final>(.*?)</final>`)

	// tagStartRe encontra o próximo início de tag reconhecida, em ordem de aparição no texto.
	tagStartRe = regexp.MustCompile(`<(think|tool_call|tool_result|final)[ >]`)
)

// Parse varre raw_text em ordem e devolve os segmentos na sequência em que aparecem —
// necessário porque uma trajetória intercala think/tool_call/tool_result várias vezes antes
// do final, e a UI precisa renderizar na ordem certa, não agrupado por tipo.
func Parse(rawText string) []Segment {
	segs, _ := ParseWithTail(rawText)
	return segs
}

// ParseWithTail é igual a Parse, mas também devolve o texto que sobrou depois do último
// segmento COMPLETO (tag fechada) — usado pela UI para mostrar, sem estilo, uma tag ainda
// sendo gerada durante o streaming, sem precisar de um parser incremental de verdade.
func ParseWithTail(rawText string) ([]Segment, string) {
	var out []Segment
	remaining := rawText

parseLoop:
	for {
		loc := tagStartRe.FindStringIndex(remaining)
		if loc == nil {
			break
		}
		remaining = remaining[loc[0]:]

		switch {
		case thinkRe.MatchString(remaining) && thinkRe.FindStringIndex(remaining)[0] == 0:
			m := thinkRe.FindStringSubmatchIndex(remaining)
			out = append(out, Segment{Kind: KindThink, Body: remaining[m[2]:m[3]]})
			remaining = remaining[m[1]:]
		case toolCallRe.MatchString(remaining) && toolCallRe.FindStringIndex(remaining)[0] == 0:
			m := toolCallRe.FindStringSubmatchIndex(remaining)
			out = append(out, Segment{
				Kind:     KindToolCall,
				ToolName: remaining[m[2]:m[3]],
				Body:     remaining[m[4]:m[5]],
			})
			remaining = remaining[m[1]:]
		case toolResultRe.MatchString(remaining) && toolResultRe.FindStringIndex(remaining)[0] == 0:
			m := toolResultRe.FindStringSubmatchIndex(remaining)
			out = append(out, Segment{
				Kind:     KindToolResult,
				ToolName: remaining[m[2]:m[3]],
				Status:   remaining[m[4]:m[5]],
				Body:     remaining[m[6]:m[7]],
			})
			remaining = remaining[m[1]:]
		case finalRe.MatchString(remaining) && finalRe.FindStringIndex(remaining)[0] == 0:
			m := finalRe.FindStringSubmatchIndex(remaining)
			out = append(out, Segment{Kind: KindFinal, Body: remaining[m[2]:m[3]]})
			remaining = remaining[m[1]:]
		default:
			// A tag abriu mas ainda não fechou (geração em andamento) — pra de tentar
			// consumir mais segmentos e devolve isso como cauda incompleta, em vez de
			// avançar char a char (o que cortaria o '<' e corromperia a cauda).
			break parseLoop
		}
	}

	return out, remaining
}
