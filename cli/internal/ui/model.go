package ui

import (
	"context"
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
	sessionTokens  int          // SOMA de todo prompt_eval_count real reportado pela Ollama nesta sessão
	promptCalls    int          // quantas chamadas reais de generate() contribuíram pra sessionTokens

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
			m.promptCalls = 0
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
		if msg.ContextTokens > 0 {
			m.sessionTokens += msg.ContextTokens
			m.promptCalls++
		}
		if msg.Delta != "" {
			m.currentRaw += msg.Delta
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

// statusLine monta o lado esquerdo do rodapé — "Ready" parado, ou spinner + tempo decorrido
// + taxa real de caracteres/s enquanto gera (nada de token/s estimado — não temos contagem
// de token de verdade nesta build pro texto gerado, só caracteres, então é isso que
// reportamos aqui; o contador de CONTEXTO do lado direito é que usa token real, ver
// contextLine).
func (m Model) statusLine() string {
	if m.streamErr != nil {
		return styleToolResultErrLabel.Render("erro: " + m.streamErr.Error())
	}
	if !m.generating {
		return "Ready"
	}

	elapsed := time.Since(m.genStarted)
	chars := len([]rune(m.currentRaw))
	rate := 0.0
	if elapsed.Seconds() > 0 {
		rate = float64(chars) / elapsed.Seconds()
	}
	return fmt.Sprintf(
		"%s gerando... %.1fs · %d caracteres · %.0f car/s",
		m.spinner.View(), elapsed.Seconds(), chars, rate,
	)
}

// contextLine monta o lado direito do rodapé — SOMA de tokens de prompt reais (não
// estimados) reportados pela própria Ollama (prompt_eval_count) em cada chamada de
// generate() feita nesta sessão. É um contador de custo/uso acumulado, não "quanto da
// janela de contexto está preenchida agora" — cada chamada reprocessa o prompt inteiro do
// zero (a Ollama não reaproveita KV cache entre requisições HTTP separadas aqui), então
// somar é a contagem honesta de quanto foi processado de verdade, não uma métrica de
// ocupação da janela (por isso não comparamos mais contra NumCtx). Persiste até Ctrl+R —
// nunca decresce nem reseta sozinho.
func (m Model) contextLine() string {
	if m.sessionTokens == 0 {
		return ""
	}
	turns := fmt.Sprintf("%d turno", len(m.history))
	if len(m.history) != 1 {
		turns += "s"
	}
	return fmt.Sprintf("%d tokens (sessão, %d chamadas) · %s · Ctrl+R reinicia", m.sessionTokens, m.promptCalls, turns)
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
		m.renderedLog = append(m.renderedLog, renderClosedTrajectory(m.currentRaw))
		m.history = append(m.history, agent.Turn{UserRequest: m.currentRequest, RawText: m.currentRaw})
		if len(m.history) > maxHistoryTurns {
			m.history = m.history[len(m.history)-maxHistoryTurns:]
		}
	}
	m.currentRaw = ""
	m.revealedLen = 0
	m.refreshViewport()
}

func (m *Model) refreshViewport() {
	visible := string([]rune(m.currentRaw)[:m.revealedLen])
	stillTyping := m.revealedLen < len([]rune(m.currentRaw)) || !m.agentDone

	content := m.renderedLog
	if visible != "" {
		content = append(append([]string{}, m.renderedLog...), renderLiveTrajectory(visible, stillTyping, m.spinner.View()))
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
// já "assentado" na cor bege base. Dois comportamentos deliberados, diferentes da renderização
// ao vivo: (1) segmentos "Thinking" são OMITIDOS aqui — só aparecem enquanto a geração está
// rolando (renderLiveTrajectory), somem do registro permanente quando a resposta termina; (2)
// usa ParseWithTail, não Parse, porque uma resposta pode ser cortada antes de fechar a última
// tag (ex.: limite de tokens) — descartar essa cauda incompleta faria o texto gerado
// desaparecer silenciosamente em vez de aparecer truncado.
func renderClosedTrajectory(rawText string) string {
	segs, tail := segments.ParseWithTail(rawText)
	var blocks []string
	for _, seg := range segs {
		if seg.Kind == segments.KindThink {
			continue
		}
		blocks = append(blocks, renderSegment(seg))
	}

	if trimmedTail := strings.TrimSpace(tail); trimmedTail != "" {
		kind, toolName, status, body, ok := detectOpenSegment(tail)
		switch {
		case ok && kind == segments.KindThink:
			// Thinking incompleto — mesma regra, não aparece no registro final.
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
// rastro de gradiente branco->bege no trecho ainda sendo gerado (tag aberta ou fragmento
// incompleto) — é o que dá o efeito de "cursor" de digitação em tempo real. `spinnerView` só é
// usado ao lado do rótulo "Thinking" enquanto ele está em aberto (a mesma animação que já
// aparecia no rodapé, agora também junto do raciocínio que está sendo gerado).
func renderLiveTrajectory(rawText string, glowing bool, spinnerView string) string {
	segs, tail := segments.ParseWithTail(rawText)
	var blocks []string
	for _, seg := range segs {
		blocks = append(blocks, renderSegment(seg))
	}

	if tail == "" {
		return strings.Join(blocks, "\n")
	}

	kind, toolName, status, body, ok := detectOpenSegment(tail)
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

	label, bodyText := labelAndBodyFor(kind, toolName, status, body)
	if glowing && kind == segments.KindThink && label != "" {
		label = spinnerView + " " + label
	}
	if glowing {
		glowed := renderGlowTail(bodyText, rgbGlowSettle)
		if label != "" {
			glowed = label + "\n" + glowed
		}
		blocks = append(blocks, glowed)
	} else {
		blocks = append(blocks, renderSegment(segments.Segment{Kind: kind, ToolName: toolName, Status: status, Body: body}))
	}

	return strings.Join(blocks, "\n")
}

func labelAndBodyFor(kind segments.Kind, toolName, status, body string) (label, bodyText string) {
	trimmed := strings.TrimSpace(body)
	switch kind {
	case segments.KindThink:
		return styleThinkLabel.Render("Thinking"), trimmed
	case segments.KindToolCall:
		return styleToolCallLabel.Render(fmt.Sprintf("tool_call · %s", toolName)), trimmed
	case segments.KindToolResult:
		if status == "ok" {
			return styleToolResultOKLabel.Render(fmt.Sprintf("tool_result · %s · ok", toolName)), trimmed
		}
		return styleToolResultErrLabel.Render(fmt.Sprintf("tool_result · %s · error", toolName)), trimmed
	case segments.KindFinal:
		return "", trimmed
	default:
		return "", trimmed
	}
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

func renderSegment(seg segments.Segment) string {
	switch seg.Kind {
	case segments.KindThink:
		return styleThinkLabel.Render("Thinking") + "\n" + styleThinkBody.Render(strings.TrimSpace(seg.Body))
	case segments.KindToolCall:
		label := styleToolCallLabel.Render(fmt.Sprintf("tool_call · %s", seg.ToolName))
		return label + "\n" + styleToolCallBody.Render(strings.TrimSpace(seg.Body))
	case segments.KindToolResult:
		if seg.Status == "ok" {
			label := styleToolResultOKLabel.Render(fmt.Sprintf("tool_result · %s · ok", seg.ToolName))
			return label + "\n" + styleToolResultBody.Render(strings.TrimSpace(seg.Body))
		}
		label := styleToolResultErrLabel.Render(fmt.Sprintf("tool_result · %s · error", seg.ToolName))
		return label + "\n" + styleToolResultErrBodyTxt.Render(strings.TrimSpace(seg.Body))
	case segments.KindFinal:
		return styleFinalBody.Render(strings.TrimSpace(seg.Body))
	default:
		return strings.TrimSpace(seg.Body)
	}
}
