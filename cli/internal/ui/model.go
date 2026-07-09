package ui

import (
	"context"
	"fmt"
	"strings"

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

type Model struct {
	viewport   viewport.Model
	textinput  textinput.Model
	spinner    spinner.Model
	client     *ollamaclient.Client
	ctx        context.Context
	streamCh   chan agent.Event
	generating bool
	streamErr  error

	renderedLog []string // trajetórias já finalizadas
	currentRaw  string   // raw_text da trajetória em andamento (streaming)

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
			m.streamErr = nil
			m.currentRaw = ""

			ch := make(chan agent.Event)
			m.streamCh = ch
			go agent.Run(m.ctx, m.client, text, ch)

			cmds = append(cmds, m.spinner.Tick, waitForEvent(ch))
		}

	case agentEventMsg:
		if msg.Err != nil {
			m.streamErr = msg.Err
			m.generating = false
			m.finalizeCurrentTrajectory()
			break
		}
		if msg.Delta != "" {
			m.currentRaw += msg.Delta
			m.refreshViewport()
		}
		if msg.Done {
			m.generating = false
			m.finalizeCurrentTrajectory()
		} else {
			cmds = append(cmds, waitForEvent(m.streamCh))
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
	block := styleUserLabel.Render("você") + "\n" + styleUserBody.Render(text)
	m.renderedLog = append(m.renderedLog, block)
	m.refreshViewport()
}

// finalizeCurrentTrajectory move o que foi acumulado durante o streaming para o histórico
// permanente e limpa o buffer de streaming — chamado quando o loop do agente termina.
func (m *Model) finalizeCurrentTrajectory() {
	if strings.TrimSpace(m.currentRaw) != "" {
		m.renderedLog = append(m.renderedLog, renderTrajectory(m.currentRaw))
	}
	m.currentRaw = ""
	m.refreshViewport()
}

func (m *Model) refreshViewport() {
	content := m.renderedLog
	if m.currentRaw != "" {
		content = append(append([]string{}, m.renderedLog...), renderTrajectory(m.currentRaw))
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

func welcomeText() string {
	return styleThinkBody.Render(
		"Sessão iniciada — conectado no Ollama local em modo raw (sem chat template).\n" +
			"Ferramentas ainda não executam de verdade nesta build (só a geração é real).\n" +
			"Digite um pedido abaixo. Ctrl+C ou Esc para sair.",
	)
}

// renderTrajectory estiliza os segmentos já FECHADOS de rawText e acrescenta, sem estilo
// (efeito de "ainda sendo digitado"), qualquer fragmento final ainda incompleto — dá o
// feedback visual de streaming em tempo real sem exigir parsing incremental de tag aberta.
func renderTrajectory(rawText string) string {
	segs, tail := segments.ParseWithTail(rawText)
	var blocks []string
	for _, seg := range segs {
		blocks = append(blocks, renderSegment(seg))
	}

	if strings.TrimSpace(tail) != "" {
		blocks = append(blocks, styleThinkBody.Render(tail))
	}

	return strings.Join(blocks, "\n")
}

func renderSegment(seg segments.Segment) string {
	switch seg.Kind {
	case segments.KindThink:
		return styleThinkLabel.Render("think") + "\n" + styleThinkBody.Render(strings.TrimSpace(seg.Body))
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
