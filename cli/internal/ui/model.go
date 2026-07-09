package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/devlucascfarias/Conatus-Logos/cli/internal/agent"
	"github.com/devlucascfarias/Conatus-Logos/cli/internal/ollamaclient"
	"github.com/devlucascfarias/Conatus-Logos/cli/internal/segments"
)

// agentEventMsg embrulha agent.Event para o loop de mensagens do bubbletea.
type agentEventMsg agent.Event

// revealTickMsg avança o efeito de digitação (rastro de gradiente) — desacoplado da
// velocidade real de chegada dos chunks de rede, pra sempre parecer uma digitação suave
// mesmo se a Ollama mandar um pedaço grande de uma vez.
type revealTickMsg struct{}

const revealInterval = 16 * time.Millisecond
const revealStepRunes = 2

// maxHistoryTurns limita quantos turnos anteriores entram no prompt — sem isso o contexto
// cresce sem parar e eventualmente estoura NumCtx (a Ollama trunca por conta própria pela
// esquerda quando isso acontece, o que pode cortar até o system prompt). Escolha
// conservadora dado NumCtx=4096 e trajetórias podendo ser longas (write_file+checker inteiro
// embutido em JSON).
const maxHistoryTurns = 6

type Model struct {
	viewport   viewport.Model
	textinput  textinput.Model
	spinner    spinner.Model
	client     *ollamaclient.Client
	workDir    string
	ctx        context.Context
	streamCh   chan agent.Event
	generating bool
	agentDone  bool
	streamErr  error

	history        []agent.Turn // turnos concluídos desta sessão (D-cli-session-history)
	currentRequest string       // pedido que iniciou o turno em andamento, pra virar Turn ao finalizar
	sessionTokens  int          // SOMA real de TotalTokens de cada turno concluído — só cresce, nunca reseta sozinho
	contextTokens  int          // última medição real de TotalTokens (pro medidor/barra do lado direito)

	thinkDurations []time.Duration // duração real de cada bloco "Thinking" já FECHADO neste turno, em ordem
	thinkIsOpen    bool
	thinkOpenSince time.Time

	renderedLog []string // trajetórias já finalizadas
	currentRaw  string   // raw_text completo recebido até agora (fonte da verdade)
	revealedLen int      // quantos runes de currentRaw já foram "digitados" na tela
	genStarted  time.Time

	width  int
	height int
	ready  bool
}

func New(client *ollamaclient.Client, workDir string) Model {
	ti := textinput.New()
	ti.Placeholder = "Digite um pedido e pressione Enter..."
	ti.Focus()
	ti.CharLimit = 2000
	ti.Prompt = "❯ "
	// A lib tem cor padrão própria (rosa/roxo) pra prompt/cursor/placeholder — sem
	// sobrescrever isso explicitamente, ela vaza por cima da paleta bege/madeira.
	ti.PromptStyle = lipgloss.NewStyle().Foreground(colorWoodLabel)
	ti.TextStyle = lipgloss.NewStyle().Foreground(colorWoodBase)
	ti.PlaceholderStyle = lipgloss.NewStyle().Foreground(colorStatus)
	ti.Cursor.Style = lipgloss.NewStyle().Foreground(colorWoodLabel)
	ti.Cursor.TextStyle = lipgloss.NewStyle().Foreground(colorWoodBase)

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = styleFinalLabel

	return Model{
		textinput: ti,
		spinner:   sp,
		client:    client,
		workDir:   workDir,
		ctx:       context.Background(),
	}
}

func (m Model) Init() tea.Cmd {
	return textinput.Blink
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height

		headerHeight := 2
		inputHeight := 3
		statusHeight := 1
		vpHeight := msg.Height - headerHeight - inputHeight - statusHeight
		if vpHeight < 3 {
			vpHeight = 3
		}

		if !m.ready {
			m.viewport = viewport.New(msg.Width-2, vpHeight)
			m.viewport.SetContent(welcomeText(m.workDir))
			m.ready = true
		} else {
			m.viewport.Width = msg.Width - 2
			m.viewport.Height = vpHeight
		}
		m.textinput.Width = msg.Width - 6

	case tea.KeyMsg:
		switch msg.Type {
		case tea.KeyCtrlC, tea.KeyEsc:
			return m, tea.Quit
		case tea.KeyCtrlR:
			if m.generating {
				break
			}
			m.history = nil
			m.sessionTokens = 0
			m.contextTokens = 0
			m.renderedLog = append(m.renderedLog, styleStatusBar.Render("— sessão reiniciada, histórico limpo —"))
			m.refreshViewport()
		case tea.KeyEnter:
			if m.generating {
				break
			}
			text := strings.TrimSpace(m.textinput.Value())
			if text == "" {
				break
			}
			m.appendUserMessage(text)
			m.textinput.Reset()
			m.generating = true
			m.agentDone = false
			m.streamErr = nil
			m.currentRequest = text
			m.currentRaw = ""
			m.revealedLen = 0
			m.thinkDurations = nil
			m.thinkIsOpen = false
			m.genStarted = time.Now()

			ch := make(chan agent.Event)
			m.streamCh = ch
			go agent.Run(m.ctx, m.client, text, m.history, m.workDir, ch)

			cmds = append(cmds, m.spinner.Tick, waitForEvent(ch), revealTick())
		}

	case agentEventMsg:
		if msg.Err != nil {
			m.streamErr = msg.Err
			m.generating = false
			m.agentDone = true
			break
		}
		if msg.TotalTokens > 0 {
			m.sessionTokens += msg.TotalTokens
			m.contextTokens = msg.TotalTokens
		}
		if msg.Delta != "" {
			m.currentRaw += msg.Delta
			m.trackThinkDuration()
		}
		if msg.Done {
			m.agentDone = true
		} else {
			cmds = append(cmds, waitForEvent(m.streamCh))
		}

	case revealTickMsg:
		total := len([]rune(m.currentRaw))
		if m.revealedLen < total {
			m.revealedLen += revealStepRunes
			if m.revealedLen > total {
				m.revealedLen = total
			}
			m.refreshViewport()
		}
		caughtUp := m.revealedLen >= total
		if caughtUp && m.agentDone {
			m.generating = false
			m.finalizeCurrentTrajectory()
		} else {
			cmds = append(cmds, revealTick())
		}

	case spinner.TickMsg:
		if m.generating {
			var cmd tea.Cmd
			m.spinner, cmd = m.spinner.Update(msg)
			cmds = append(cmds, cmd)
		}
	}

	var tiCmd, vpCmd tea.Cmd
	m.textinput, tiCmd = m.textinput.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)
	cmds = append(cmds, tiCmd, vpCmd)

	return m, tea.Batch(cmds...)
}

func (m Model) View() string {
	if !m.ready {
		return "iniciando..."
	}

	header := styleTitle.Render(appTitle(m.client, m.workDir))

	left := m.statusLine()
	right := m.contextLine()
	gap := m.width - 2 - lipgloss.Width(left) - lipgloss.Width(right) - 2 // -2 do Padding(0,1) da própria barra
	if gap < 1 {
		gap = 1
	}
	statusBar := styleStatusBar.Render(left + strings.Repeat(" ", gap) + right)

	body := styleViewport.Width(m.width - 2).Render(m.viewport.View())
	input := styleInputBox.Width(m.width - 2).Render(m.textinput.View())

	return lipglossJoinVertical(header, body, input, statusBar)
}

// statusLine monta o lado esquerdo do rodapé — o contador de tokens da SESSÃO (soma real,
// medida via ollamaclient.CountTokens uma vez por turno concluído — ver agent.finishTurn)
// nunca desaparece nem reseta sozinho, só cresce até Ctrl+R. Durante a geração do turno
// atual, o número mostrado ainda é o total ANTES desse turno (a medição só fecha quando o
// turno termina) — por isso o spinner+tempo decorrido ao lado deixa claro que ainda está em
// andamento, em vez de fingir uma contagem ao vivo que não temos de verdade.
func (m Model) statusLine() string {
	if m.streamErr != nil {
		return styleToolResultErrLabel.Render("erro: " + m.streamErr.Error())
	}

	tokensText := fmt.Sprintf("%d tokens", m.sessionTokens)
	if !m.generating {
		return "Ready · " + tokensText
	}

	elapsed := time.Since(m.genStarted)
	return fmt.Sprintf("%s gerando... %.1fs · %s", m.spinner.View(), elapsed.Seconds(), tokensText)
}

// contextLine monta o lado direito do rodapé — a última medição REAL (ollamaclient.
// CountTokens, não estimada) de quantos tokens o prompt completo (system + histórico +
// turno) ocupa, como uma barra preenchendo até NumCtx. Diferente de statusLine (que soma
// tudo), aqui é um medidor de OCUPAÇÃO ATUAL — pode até encolher entre turnos se
// maxHistoryTurns descartar turnos antigos do histórico. Fica vazio até a primeira medição
// chegar (fim do primeiro turno).
func (m Model) contextLine() string {
	if m.contextTokens == 0 {
		return ""
	}
	pct := float64(m.contextTokens) / float64(m.client.NumCtx) * 100
	if pct > 100 {
		pct = 100
	}
	return fmt.Sprintf("Context: %s %d/%d (%.0f%%)", renderContextBar(pct, contextBarWidth), m.contextTokens, m.client.NumCtx, pct)
}

const contextBarWidth = 16

func renderContextBar(pct float64, width int) string {
	filled := int(pct/100*float64(width) + 0.5)
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	filledStr := lipgloss.NewStyle().Foreground(colorOKGold).Render(strings.Repeat("█", filled))
	emptyStr := lipgloss.NewStyle().Foreground(colorBorderCol).Render(strings.Repeat("░", width-filled))
	return filledStr + emptyStr
}

// trackThinkDuration detecta quando um bloco "Thinking" abre e fecha DE VERDADE (comparando
// o fragmento ainda aberto — ParseWithTail — a cada novo pedaço de texto recebido) e grava a
// duração real em thinkDurations quando ele fecha. Chamado a cada Delta recebido do agente,
// não a cada tick de revelação — a duração reflete o tempo real de geração, não a velocidade
// da animação de digitação na tela.
func (m *Model) trackThinkDuration() {
	_, tail := segments.ParseWithTail(m.currentRaw)
	kind, _, _, _, ok := detectOpenSegment(tail)
	isThinkOpenNow := ok && kind == segments.KindThink

	if isThinkOpenNow && !m.thinkIsOpen {
		m.thinkIsOpen = true
		m.thinkOpenSince = time.Now()
	} else if !isThinkOpenNow && m.thinkIsOpen {
		m.thinkIsOpen = false
		m.thinkDurations = append(m.thinkDurations, time.Since(m.thinkOpenSince))
	}
}

func appTitle(client *ollamaclient.Client, workDir string) string {
	return fmt.Sprintf("Conatus — %s (%s) — %s", client.Model, client.Host, workDir)
}

func (m *Model) appendUserMessage(text string) {
	m.renderedLog = append(m.renderedLog, styleUserBody.Render(text))
	m.refreshViewport()
}

// finalizeCurrentTrajectory move o que foi acumulado durante o streaming para o histórico
// VISUAL permanente (já totalmente "assentado", sem gradiente) e limpa o buffer — chamado
// quando o loop do agente termina E a animação de digitação termina de alcançar o texto
// real. Também grava o turno no histórico de SESSÃO (agent.Turn) usado nos próximos pedidos
// como contexto — a trajetória INTEIRA entra aqui, não só a resposta final, porque é esse o
// formato que o [ASSISTANT] sempre teve durante o treino (ver aviso em agent.go).
func (m *Model) finalizeCurrentTrajectory() {
	if strings.TrimSpace(m.currentRaw) != "" {
		m.renderedLog = append(m.renderedLog, renderClosedTrajectory(m.currentRaw, m.thinkDurations))
		m.history = append(m.history, agent.Turn{UserRequest: m.currentRequest, RawText: m.currentRaw})
		if len(m.history) > maxHistoryTurns {
			m.history = m.history[len(m.history)-maxHistoryTurns:]
		}
	}
	m.currentRaw = ""
	m.revealedLen = 0
	m.thinkDurations = nil
	m.thinkIsOpen = false
	m.refreshViewport()
}

func (m *Model) refreshViewport() {
	visible := string([]rune(m.currentRaw)[:m.revealedLen])
	stillTyping := m.revealedLen < len([]rune(m.currentRaw)) || !m.agentDone

	content := m.renderedLog
	if visible != "" {
		content = append(append([]string{}, m.renderedLog...), renderLiveTrajectory(visible, stillTyping, m.spinner.View(), m.thinkDurations))
	}

	joined := strings.Join(content, "\n\n")
	// viewport da bubbles NÃO quebra linha sozinho — sem isso, qualquer linha mais longa
	// que a largura da caixa fica cortada até a janela ser redimensionada (o redraw força
	// um recálculo). O Width() do lipgloss é ANSI-aware, não estraga as cores já aplicadas.
	if m.viewport.Width > 0 {
		joined = lipgloss.NewStyle().Width(m.viewport.Width).Render(joined)
	}
	m.viewport.SetContent(joined)
	m.viewport.GotoBottom()
}

func waitForEvent(ch chan agent.Event) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return agentEventMsg{Done: true}
		}
		return agentEventMsg(ev)
	}
}

func revealTick() tea.Cmd {
	return tea.Tick(revealInterval, func(time.Time) tea.Msg { return revealTickMsg{} })
}

func welcomeText(workDir string) string {
	return styleThinkBody.Render(
		"Sessão iniciada — conectado no Ollama local em modo raw (sem chat template).\n" +
			"write_file/read_file/list_files executam de verdade em " + workDir + ".\n" +
			"checker/shell/web_search ainda não estão portados (respondem UNSUPPORTED_TOOL).\n" +
			"Os últimos turnos ficam no contexto dos próximos pedidos (histórico não é um formato " +
			"treinado — pode não funcionar tão bem quanto um pedido isolado).\n" +
			"Digite um pedido abaixo. Ctrl+R reinicia a sessão (limpa o histórico). Ctrl+C ou Esc para sair.",
	)
}

// renderClosedTrajectory estiliza uma trajetória totalmente finalizada — sem gradiente, tudo
// já "assentado" na cor bege base. Blocos "Thinking" viram "Thought for Xs" (duração real,
// de thinkDurations — ver trackThinkDuration), nunca mostrando o raciocínio em si, mas
// também nunca desaparecendo por completo. Usa ParseWithTail, não Parse, porque uma resposta
// pode ser cortada antes de fechar a última tag (ex.: limite de tokens) — descartar essa
// cauda incompleta faria o texto gerado desaparecer silenciosamente em vez de aparecer
// truncado.
func renderClosedTrajectory(rawText string, thinkDurations []time.Duration) string {
	segs, tail := segments.ParseWithTail(rawText)
	blocks := renderCollapsedSegments(segs, thinkDurations, "")

	if trimmedTail := strings.TrimSpace(tail); trimmedTail != "" {
		kind, toolName, status, body, ok := detectOpenSegment(tail)
		switch {
		case ok && kind == segments.KindThink:
			// Thinking cortado antes de fechar — sem duração final conhecida, omite.
		case ok && kind == segments.KindToolCall:
			// resposta cortada bem no meio de uma chamada de ferramenta — sem resultado
			// pra parear, não dá pra saber se passou ou falhou.
			blocks = append(blocks, styleToolResultErrLabel.Render("⋯ "+toolName+" (cortado antes do resultado)"))
		case ok && kind == segments.KindToolResult:
			blocks = append(blocks, renderToolResultLine(toolName, status, body))
		case ok:
			blocks = append(blocks, renderSegment(segments.Segment{Kind: kind, ToolName: toolName, Status: status, Body: body}))
		default:
			// Fragmento cru não reconhecido (cortou no meio de uma tag) — mostra mesmo
			// assim, sem rótulo, pra não perder conteúdo que o modelo realmente gerou.
			blocks = append(blocks, styleFinalBody.Render(trimmedTail))
		}
	}

	return strings.Join(blocks, "\n")
}

// renderLiveTrajectory renderiza segmentos já fechados normalmente e, se `glowing`, aplica o
// rastro de gradiente branco->bege no trecho de think/final ainda sendo gerado — é o que dá
// o efeito de "cursor" de digitação em tempo real. Chamadas de ferramenta NUNCA mostram o
// JSON dos argumentos/resultado (D-cli-tool-connector-view): enquanto não há
// `tool_result` pareado ainda (a ferramenta está executando de verdade — pode levar
// segundos num `checker` com pytest), mostra spinner + nome, animado; assim que o resultado
// chega, vira uma linha estática "└ nome" (dourado se ok, rust se erro, com a mensagem de
// erro real embaixo quando aplicável). Blocos "Thinking" já fechados viram "Thought for Xs".
func renderLiveTrajectory(rawText string, glowing bool, spinnerView string, thinkDurations []time.Duration) string {
	segs, tail := segments.ParseWithTail(rawText)
	blocks := renderCollapsedSegments(segs, thinkDurations, spinnerView)

	if tail == "" {
		return strings.Join(blocks, "\n")
	}

	kind, toolName, _, body, ok := detectOpenSegment(tail)
	if !ok {
		// Ainda digitando a própria tag de abertura (ex.: "<thi") — não dá pra saber o
		// tipo ainda, mostra só o rastro cru por um instante muito breve.
		if glowing {
			blocks = append(blocks, renderGlowTail(tail, rgbGlowSettle))
		} else {
			blocks = append(blocks, styleThinkBody.Render(tail))
		}
		return strings.Join(blocks, "\n")
	}

	bodyText := strings.TrimSpace(body)
	switch kind {
	case segments.KindToolCall:
		// Nome da ferramenta já reconhecido (achou o `">` de fechamento do atributo
		// name), mesmo que os argumentos ainda estejam sendo digitados — nunca mostra
		// esse JSON, só o indicador animado.
		blocks = append(blocks, renderToolPendingLine(toolName, spinnerView))
	case segments.KindThink:
		label := spinnerView + " " + styleThinkLabel.Render("Thinking")
		if glowing {
			blocks = append(blocks, label+"\n"+renderGlowTail(bodyText, rgbGlowSettle))
		} else {
			blocks = append(blocks, label+"\n"+styleThinkBody.Render(bodyText))
		}
	case segments.KindFinal:
		if glowing {
			blocks = append(blocks, renderGlowTail(bodyText, rgbGlowSettle))
		} else {
			blocks = append(blocks, styleFinalBody.Render(bodyText))
		}
	default:
		// tool_result nunca deveria chegar parcialmente formado (é montado inteiro pelo
		// nosso próprio código, não gerado token a token pelo modelo), mas não trava se
		// acontecer.
		blocks = append(blocks, renderToolPendingLine(toolName, spinnerView))
	}

	return strings.Join(blocks, "\n")
}

// renderCollapsedSegments varre os segmentos JÁ FECHADOS em ordem e funde cada par
// tool_call+tool_result adjacente numa única linha compacta — nunca mostra o JSON de
// argumentos nem o corpo bruto do resultado. Um tool_call sem o tool_result logo em
// seguida significa que a ferramenta ainda está executando de verdade (spinnerView anima
// isso); só acontece na visão ao vivo, nunca numa trajetória já finalizada. Cada bloco
// "Thinking" fechado vira "Thought for Xs" usando thinkDurations, na ordem em que os blocos
// de pensamento aparecem (thinkIdx acompanha isso).
func renderCollapsedSegments(segs []segments.Segment, thinkDurations []time.Duration, spinnerView string) []string {
	var blocks []string
	thinkIdx := 0
	for i := 0; i < len(segs); i++ {
		seg := segs[i]
		switch seg.Kind {
		case segments.KindThink:
			var dur time.Duration
			if thinkIdx < len(thinkDurations) {
				dur = thinkDurations[thinkIdx]
			}
			blocks = append(blocks, renderThoughtSummary(dur))
			thinkIdx++
		case segments.KindToolCall:
			if i+1 < len(segs) && segs[i+1].Kind == segments.KindToolResult {
				next := segs[i+1]
				blocks = append(blocks, renderToolResultLine(next.ToolName, next.Status, next.Body))
				i++
			} else {
				blocks = append(blocks, renderToolPendingLine(seg.ToolName, spinnerView))
			}
		case segments.KindToolResult:
			// Órfão (não deveria acontecer — todo tool_result vem logo após seu
			// tool_call), mas renderiza igual pra não perder informação real.
			blocks = append(blocks, renderToolResultLine(seg.ToolName, seg.Status, seg.Body))
		default:
			blocks = append(blocks, renderSegment(seg))
		}
	}
	return blocks
}

// renderToolPendingLine é o estado ANIMADO — ferramenta chamada, resultado ainda não
// chegou (execução real em andamento, ex.: checker rodando pytest de verdade).
func renderToolPendingLine(toolName, spinnerView string) string {
	return spinnerView + " " + styleToolCallLabel.Render(toolName)
}

// renderThoughtSummary é o estado ESTÁTICO de um bloco Thinking já fechado — nunca mostra o
// texto do raciocínio, só quanto tempo real ele levou. `d == 0` significa que a duração não
// foi capturada (não deveria acontecer em uso normal — trackThinkDuration grava toda vez que
// um bloco fecha —, mas evita "Thought for 0s" caso aconteça).
func renderThoughtSummary(d time.Duration) string {
	if d <= 0 {
		return styleThinkLabel.Render("Thought")
	}
	return styleThinkLabel.Render(fmt.Sprintf("Thought for %.0fs", d.Seconds()))
}

// renderToolResultLine é o estado ESTÁTICO — resultado já chegou. Conector "└" fixo, sem
// spinner. Nunca mostra o corpo bruto do resultado; pra erro, extrai só a mensagem real
// (extractErrorMessage) pra continuar útil sem virar um dump de JSON.
func renderToolResultLine(toolName, status, body string) string {
	if status == "ok" {
		return "└ " + styleToolResultOKLabel.Render(toolName)
	}
	line := "└ " + styleToolResultErrLabel.Render(toolName)
	if msg := extractErrorMessage(body); msg != "" {
		line += "\n" + styleToolResultErrBodyTxt.Render("  "+msg)
	}
	return line
}

// extractErrorMessage tenta achar a mensagem de erro real dentro do JSON do tool_result —
// aceita tanto o formato simples ({"code","message"}, usado por write_file/read_file/
// list_files/UNSUPPORTED_TOOL) quanto o formato do checker ({"errors":[{"message":...}]}).
// Nunca inventa uma mensagem — se não achar nenhum campo reconhecido, devolve "".
func extractErrorMessage(body string) string {
	var parsed map[string]any
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		return ""
	}
	if msg, ok := parsed["message"].(string); ok && msg != "" {
		return msg
	}
	if rawErrors, ok := parsed["errors"].([]any); ok && len(rawErrors) > 0 {
		if first, ok := rawErrors[0].(map[string]any); ok {
			if msg, ok := first["message"].(string); ok {
				return msg
			}
		}
	}
	return ""
}

// detectOpenSegment reconhece o tipo de uma tag que já ABRIU mas ainda não fechou — permite
// mostrar o rótulo certo (ex.: "Thinking") assim que a tag de abertura é reconhecida, sem
// esperar a tag de fechamento chegar.
func detectOpenSegment(tail string) (kind segments.Kind, toolName, status, body string, ok bool) {
	switch {
	case strings.HasPrefix(tail, "<think>"):
		return segments.KindThink, "", "", tail[len("<think>"):], true
	case strings.HasPrefix(tail, "<final>"):
		return segments.KindFinal, "", "", tail[len("<final>"):], true
	case strings.HasPrefix(tail, `<tool_call name="`):
		rest := tail[len(`<tool_call name="`):]
		idx := strings.Index(rest, `">`)
		if idx < 0 {
			return 0, "", "", "", false
		}
		return segments.KindToolCall, rest[:idx], "", rest[idx+2:], true
	case strings.HasPrefix(tail, `<tool_result name="`):
		rest := tail[len(`<tool_result name="`):]
		nameEnd := strings.Index(rest, `" status="`)
		if nameEnd < 0 {
			return 0, "", "", "", false
		}
		name := rest[:nameEnd]
		rest2 := rest[nameEnd+len(`" status="`):]
		statusEnd := strings.Index(rest2, `">`)
		if statusEnd < 0 {
			return 0, "", "", "", false
		}
		return segments.KindToolResult, name, rest2[:statusEnd], rest2[statusEnd+2:], true
	default:
		return 0, "", "", "", false
	}
}

// renderSegment só lida com Final agora — Think vira "Thought for Xs" via renderThoughtSummary
// e tool_call/tool_result são sempre interceptados antes por renderCollapsedSegments (vira
// "└ nome"/spinner, nunca JSON cru).
func renderSegment(seg segments.Segment) string {
	switch seg.Kind {
	case segments.KindFinal:
		return renderFinalBody(strings.TrimSpace(seg.Body))
	default:
		return strings.TrimSpace(seg.Body)
	}
}
