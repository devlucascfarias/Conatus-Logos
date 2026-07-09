// renderização de blocos de código markdown (```lang ... ```) dentro do corpo de um <final> —
// destaque de sintaxe real via chroma (não uma paleta fixa de palavras-chave escrita à mão,
// que ficaria errada/incompleta pra qualquer linguagem fora de umas poucas) + coluna de
// número de linha, no mesmo espírito de "nunca mostrar JSON cru" já aplicado a tool_call/
// tool_result: aqui o objetivo é nunca mostrar código sem contexto de linha/sintaxe quando o
// modelo devolve um bloco formatado.
package ui

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
)

// renderFinalBody separa texto normal de blocos de código markdown dentro do corpo de um
// <final> e renderiza cada um com o tratamento certo — texto continua em styleFinalBody,
// código ganha números de linha + destaque de sintaxe real (renderCodeBlock).
func renderFinalBody(body string) string {
	var out []string
	for _, part := range splitCodeFences(body) {
		if part.isCode {
			if block := renderCodeBlock(part.lang, part.code); block != "" {
				out = append(out, block)
			}
			continue
		}
		if trimmed := strings.TrimSpace(part.code); trimmed != "" {
			out = append(out, styleFinalBody.Render(trimmed))
		}
	}
	return strings.Join(out, "\n")
}

type textPart struct {
	isCode bool
	lang   string
	code   string
}

// splitCodeFences varre o texto linha a linha procurando cercas ```lang / ``` — um bloco de
// código que nunca fecha (resposta cortada, ex.: limite de tokens) ainda é devolvido como
// código em vez de descartado, mesmo espírito de renderClosedTrajectory de nunca perder
// conteúdo real gerado.
func splitCodeFences(text string) []textPart {
	var parts []textPart
	lines := strings.Split(text, "\n")

	var textBuf strings.Builder
	var codeBuf strings.Builder
	inCode := false
	lang := ""

	flushText := func() {
		if textBuf.Len() > 0 {
			parts = append(parts, textPart{code: textBuf.String()})
			textBuf.Reset()
		}
	}

	for _, rawLine := range lines {
		line := strings.TrimRight(rawLine, "\r")
		if fenceLang, isFence := parseFence(line); isFence {
			if !inCode {
				flushText()
				inCode = true
				lang = fenceLang
				codeBuf.Reset()
			} else {
				parts = append(parts, textPart{isCode: true, lang: lang, code: strings.TrimSuffix(codeBuf.String(), "\n")})
				inCode = false
			}
			continue
		}
		if inCode {
			codeBuf.WriteString(line)
			codeBuf.WriteString("\n")
		} else {
			textBuf.WriteString(line)
			textBuf.WriteString("\n")
		}
	}

	if inCode {
		parts = append(parts, textPart{isCode: true, lang: lang, code: strings.TrimSuffix(codeBuf.String(), "\n")})
	} else {
		flushText()
	}
	return parts
}

func parseFence(line string) (lang string, ok bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "```") {
		return "", false
	}
	return strings.TrimSpace(strings.TrimPrefix(trimmed, "```")), true
}

// renderCodeBlock aplica destaque de sintaxe real (chroma — lexer pelo nome da linguagem da
// cerca markdown, ou por análise do conteúdo se a cerca não disser qual é) e prefixa cada
// linha com seu número real, alinhado à largura do maior número do bloco.
func renderCodeBlock(lang, code string) string {
	if strings.TrimSpace(code) == "" {
		return ""
	}

	lines, err := highlightLines(lang, code)
	if err != nil {
		lines = strings.Split(code, "\n")
	}

	gutterWidth := len(fmt.Sprintf("%d", len(lines)))
	var b strings.Builder
	if lang != "" {
		b.WriteString(styleCodeLangTag.Render(lang))
		b.WriteString("\n")
	}
	for i, line := range lines {
		b.WriteString(styleCodeGutter.Render(fmt.Sprintf("%*d │ ", gutterWidth, i+1)))
		b.WriteString(line)
		if i < len(lines)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// highlightLines devolve o código já colorido (sequências ANSI reais, não uma paleta de
// palavras-chave escrita à mão) linha por linha, pronto pra receber o gutter na frente de
// cada uma. lexers.Analyse cobre o caso em que a cerca markdown não diz a linguagem.
func highlightLines(lang, code string) ([]string, error) {
	lexer := lexers.Get(lang)
	if lexer == nil {
		lexer = lexers.Analyse(code)
	}
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)

	style := styles.Get("monokai")
	if style == nil {
		style = styles.Fallback
	}

	iterator, err := lexer.Tokenise(nil, code)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	if err := formatters.TTY16m.Format(&buf, style, iterator); err != nil {
		return nil, err
	}
	return strings.Split(strings.TrimSuffix(buf.String(), "\n"), "\n"), nil
}
