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

type Model struct {
	viewport   viewport.Model
	textinput  textinput.Model
	spinner    spinner.Model
	client     *ollamaclient.Client
	ctx        context.Context
	streamCh   chan agent.Event
	generating bool
	agentDone  bool
	streamErr  error

	renderedLog []string // trajetórias já finalizadas
	currentRaw  string   // raw_text completo recebido até agora (fonte da verdade)
	revealedLen int      // quantos runes de currentRaw já foram "digitados" na tela

	width  int
	height int
	ready  bool
}

func New(client *ollamaclient.Client) Model {
	ti := textinput.New()
	ti.Placeholder = "Digite um pedido e pressione Enter..."
	ti.Focus()
	ti.CharLimit = 2000
	ti.Prompt = "❯ "

	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = styleFinalLabel

	return Model{
		textinput: ti,
		spinner:   sp,
		client:    client,
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
			m.viewport.SetContent(welcomeText())
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
			m.currentRaw = ""
			m.revealedLen = 0

			ch := make(chan agent.Event)
			m.streamCh = ch
			go agent.Run(m.ctx, m.client, text, ch)

			cmds = append(cmds, m.spinner.Tick, waitForEvent(ch), revealTick())
		}

	case agentEventMsg:
		if msg.Err != nil {
			m.streamErr = msg.Err
			m.generating = false
			m.agentDone = true
			break
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

	header := styleTitle.Render(appTitle(m.client))

	status := "pronto"
	if m.generating {
		status = m.spinner.View() + " gerando..."
	}
	if m.streamErr != nil {
		status = styleToolResultErrLabel.Render("erro: " + m.streamErr.Error())
	}
	statusBar := styleStatusBar.Render(status)

	body := styleViewport.Width(m.width - 2).Render(m.viewport.View())
	input := styleInputBox.Width(m.width - 2).Render(m.textinput.View())

	return lipglossJoinVertical(header, body, input, statusBar)
}

func appTitle(client *ollamaclient.Client) string {
	return fmt.Sprintf("Conatus-Logos — %s (%s)", client.Model, client.Host)
}

func (m *Model) appendUserMessage(text string) {
	m.renderedLog = append(m.renderedLog, styleUserBody.Render(text))
	m.refreshViewport()
}

// finalizeCurrentTrajectory move o que foi acumulado durante o streaming para o histórico
// permanente (já totalmente "assentado", sem gradiente) e limpa o buffer — chamado quando o
// loop do agente termina E a animação de digitação termina de alcançar o texto real.
func (m *Model) finalizeCurrentTrajectory() {
	if strings.TrimSpace(m.currentRaw) != "" {
		m.renderedLog = append(m.renderedLog, renderClosedTrajectory(m.currentRaw))
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
		content = append(append([]string{}, m.renderedLog...), renderLiveTrajectory(visible, stillTyping))
	}
	m.viewport.SetContent(strings.Join(content, "\n\n"))
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

func welcomeText() string {
	return styleThinkBody.Render(
		"Sessão iniciada — conectado no Ollama local em modo raw (sem chat template).\n" +
			"Ferramentas ainda não executam de verdade nesta build (só a geração é real).\n" +
			"Digite um pedido abaixo. Ctrl+C ou Esc para sair.",
	)
}

// renderClosedTrajectory estiliza uma trajetória totalmente finalizada — sem gradiente, tudo
// já "assentado" na cor bege base.
func renderClosedTrajectory(rawText string) string {
	segs := segments.Parse(rawText)
	var blocks []string
	for _, seg := range segs {
		blocks = append(blocks, renderSegment(seg))
	}
	return strings.Join(blocks, "\n")
}

// renderLiveTrajectory renderiza segmentos já fechados normalmente e, se `glowing`, aplica o
// rastro de gradiente branco->bege no trecho ainda sendo gerado (tag aberta ou fragmento
// incompleto) — é o que dá o efeito de "cursor" de digitação em tempo real.
func renderLiveTrajectory(rawText string, glowing bool) string {
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
			blocks = append(blocks, renderGlowTail(tail, rgbWoodBase))
		} else {
			blocks = append(blocks, styleThinkBody.Render(tail))
		}
		return strings.Join(blocks, "\n")
	}

	label, bodyText := labelAndBodyFor(kind, toolName, status, body)
	if glowing {
		blocks = append(blocks, label+"\n"+renderGlowTail(bodyText, rgbWoodBase))
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
		return styleFinalLabel.Render("resposta"), trimmed
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
		return styleFinalLabel.Render("resposta") + "\n" + styleFinalBody.Render(strings.TrimSpace(seg.Body))
	default:
		return strings.TrimSpace(seg.Body)
	}
}
