package ui

import (
	"strings"
	"testing"
)

func TestRenderFinalBodyAddsLineNumbersToFencedCodeBlock(t *testing.T) {
	body := "aqui está:\n```python\ndef f():\n    return 1\n```\nprontinho"
	out := renderFinalBody(body)

	if !strings.Contains(out, "1 │") || !strings.Contains(out, "2 │") {
		t.Errorf("esperava números de linha 1 e 2 no bloco de código: %q", out)
	}
	if !strings.Contains(out, "aqui está:") || !strings.Contains(out, "prontinho") {
		t.Errorf("texto normal ao redor do bloco não deveria sumir: %q", out)
	}
}

func TestRenderFinalBodyHighlightsSyntaxWithAnsiCodes(t *testing.T) {
	body := "```python\ndef f():\n    return 1\n```"
	out := renderFinalBody(body)

	if !strings.Contains(out, "\x1b[") {
		t.Errorf("esperava sequências ANSI de destaque de sintaxe no código, veio texto cru: %q", out)
	}
}

func TestRenderFinalBodyPreservesUnclosedCodeFence(t *testing.T) {
	// bloco cortado antes de fechar (limite de tokens) — não pode desaparecer.
	body := "```go\nfunc main() {\n    fmt.Println(\"oi\")"
	out := renderFinalBody(body)

	if !strings.Contains(out, "Println") {
		t.Errorf("conteúdo do bloco de código cortado foi perdido: %q", out)
	}
}

func TestRenderFinalBodyWithoutCodeFenceRendersPlainText(t *testing.T) {
	body := "só uma resposta normal, sem código nenhum"
	out := renderFinalBody(body)

	if !strings.Contains(out, "só uma resposta normal") {
		t.Errorf("texto simples sumiu: %q", out)
	}
	if strings.Contains(out, "│") {
		t.Errorf("não deveria ter gutter de código quando não há bloco de código: %q", out)
	}
}

func TestSplitCodeFencesSeparatesTextAndCodeInOrder(t *testing.T) {
	parts := splitCodeFences("antes\n```python\nx = 1\n```\ndepois")
	if len(parts) != 3 {
		t.Fatalf("esperava 3 partes (texto, código, texto), veio %d: %+v", len(parts), parts)
	}
	if parts[0].isCode || !strings.Contains(parts[0].code, "antes") {
		t.Errorf("primeira parte deveria ser texto 'antes': %+v", parts[0])
	}
	if !parts[1].isCode || parts[1].lang != "python" || !strings.Contains(parts[1].code, "x = 1") {
		t.Errorf("segunda parte deveria ser código python 'x = 1': %+v", parts[1])
	}
	if parts[2].isCode || !strings.Contains(parts[2].code, "depois") {
		t.Errorf("terceira parte deveria ser texto 'depois': %+v", parts[2])
	}
}
