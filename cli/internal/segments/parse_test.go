package segments

import "testing"

func TestParseOrdersSegmentsAsTheyAppear(t *testing.T) {
	raw := `<think>a</think><tool_call name="write_file">{}</tool_call><tool_result name="write_file" status="ok">{}</tool_result><think>b</think><final>c</final>`
	segs := Parse(raw)

	if len(segs) != 5 {
		t.Fatalf("esperava 5 segmentos, veio %d: %+v", len(segs), segs)
	}
	wantKinds := []Kind{KindThink, KindToolCall, KindToolResult, KindThink, KindFinal}
	for i, want := range wantKinds {
		if segs[i].Kind != want {
			t.Errorf("segmento %d: esperava kind %v, veio %v", i, want, segs[i].Kind)
		}
	}
}

func TestParseExtractsToolCallNameAndBody(t *testing.T) {
	raw := `<tool_call name="checker">{"language": "python"}</tool_call>`
	segs := Parse(raw)

	if len(segs) != 1 {
		t.Fatalf("esperava 1 segmento, veio %d", len(segs))
	}
	if segs[0].ToolName != "checker" {
		t.Errorf("esperava ToolName=checker, veio %q", segs[0].ToolName)
	}
	if segs[0].Body != `{"language": "python"}` {
		t.Errorf("body inesperado: %q", segs[0].Body)
	}
}

func TestParseExtractsToolResultStatus(t *testing.T) {
	raw := `<tool_result name="checker" status="error">{"code": "X"}</tool_result>`
	segs := Parse(raw)

	if len(segs) != 1 {
		t.Fatalf("esperava 1 segmento, veio %d", len(segs))
	}
	if segs[0].Status != "error" {
		t.Errorf("esperava Status=error, veio %q", segs[0].Status)
	}
}

func TestParseWithTailReturnsIncompleteTrailingTag(t *testing.T) {
	// Simula o meio de um streaming: a tag <think> abriu mas ainda não fechou.
	raw := `<final>ok</final><think>ainda gerando isso aqui`
	segs, tail := ParseWithTail(raw)

	if len(segs) != 1 || segs[0].Kind != KindFinal {
		t.Fatalf("esperava 1 segmento final fechado, veio %+v", segs)
	}
	if tail != `<think>ainda gerando isso aqui` {
		t.Errorf("cauda incompleta incorreta: %q", tail)
	}
}

func TestParseWithTailReturnsEmptyTailWhenEverythingCloses(t *testing.T) {
	raw := `<think>oi</think><final>tudo bem</final>`
	_, tail := ParseWithTail(raw)

	if tail != "" {
		t.Errorf("esperava cauda vazia (tudo fechado), veio %q", tail)
	}
}

func TestParseHandlesEmptyInput(t *testing.T) {
	segs := Parse("")
	if len(segs) != 0 {
		t.Errorf("esperava 0 segmentos para entrada vazia, veio %d", len(segs))
	}
}

func TestParseIgnoresPlainTextOutsideTags(t *testing.T) {
	// Texto solto antes/depois de tags não deve virar segmento nem quebrar o parsing.
	raw := `algum texto solto <final>resposta</final> mais texto solto`
	segs := Parse(raw)

	if len(segs) != 1 || segs[0].Kind != KindFinal {
		t.Fatalf("esperava 1 segmento final, veio %+v", segs)
	}
	if segs[0].Body != "resposta" {
		t.Errorf("body inesperado: %q", segs[0].Body)
	}
}
