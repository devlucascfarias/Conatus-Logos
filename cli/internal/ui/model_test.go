package ui

import "strings"
import "testing"

func TestDetectOpenSegmentRecognizesThinkAsSoonAsOpeningTagCloses(t *testing.T) {
	kind, _, _, body, ok := detectOpenSegment("<think>ainda gerando")
	if !ok {
		t.Fatal("esperava reconhecer think em andamento")
	}
	if kind != 0 { // KindThink == 0 (iota)
		t.Errorf("kind incorreto: %v", kind)
	}
	if body != "ainda gerando" {
		t.Errorf("body incorreto: %q", body)
	}
}

func TestDetectOpenSegmentReturnsFalseForIncompleteOpeningTag(t *testing.T) {
	_, _, _, _, ok := detectOpenSegment("<thi")
	if ok {
		t.Error("não deveria reconhecer uma tag de abertura ainda incompleta")
	}
}

func TestDetectOpenSegmentExtractsToolCallNameBeforeItCloses(t *testing.T) {
	kind, name, _, body, ok := detectOpenSegment(`<tool_call name="checker">{"lang`)
	if !ok {
		t.Fatal("esperava reconhecer tool_call assim que o atributo name fecha")
	}
	if kind != 1 { // KindToolCall
		t.Errorf("kind incorreto: %v", kind)
	}
	if name != "checker" {
		t.Errorf("nome da ferramenta incorreto: %q", name)
	}
	if body != `{"lang` {
		t.Errorf("body incorreto: %q", body)
	}
}

func TestRenderClosedTrajectoryOmitsThinkingFromFinalRecord(t *testing.T) {
	raw := `<think>vou fazer isso</think><final>pronto</final>`
	out := renderClosedTrajectory(raw)

	if strings.Contains(out, "Thinking") {
		t.Errorf("Thinking não deveria aparecer no registro finalizado: %q", out)
	}
	if !strings.Contains(out, "pronto") {
		t.Errorf("texto da resposta final sumiu: %q", out)
	}
}

func TestRenderClosedTrajectoryOmitsFinalLabel(t *testing.T) {
	raw := `<final>a resposta</final>`
	out := renderClosedTrajectory(raw)

	// A saída não deve conter uma linha só com o rótulo "resposta" separado do texto —
	// só o styleFinalBody.Render(texto), sem prefixo de rótulo.
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) == "resposta" {
			t.Errorf("rótulo 'resposta' não deveria mais aparecer: %q", out)
		}
	}
	if !strings.Contains(out, "a resposta") {
		t.Errorf("texto da resposta sumiu: %q", out)
	}
}

func TestRenderClosedTrajectoryPreservesUnclosedTrailingContent(t *testing.T) {
	// Simula uma resposta cortada (ex.: limite de tokens) antes de </final> fechar — o
	// texto gerado não pode desaparecer silenciosamente.
	raw := `<final>o código é: print('oi'` // nunca fecha
	out := renderClosedTrajectory(raw)

	if !strings.Contains(out, "print('oi'") {
		t.Errorf("conteúdo cortado foi perdido em vez de aparecido truncado: %q", out)
	}
}

func TestRenderClosedTrajectoryOmitsUnclosedThinking(t *testing.T) {
	raw := `<think>pensando e cortou aqui`
	out := renderClosedTrajectory(raw)

	if strings.Contains(out, "pensando e cortou aqui") {
		t.Errorf("Thinking incompleto não deveria aparecer no registro finalizado: %q", out)
	}
}
