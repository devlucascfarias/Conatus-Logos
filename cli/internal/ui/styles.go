package ui

import "github.com/charmbracelet/lipgloss"

// Rótulos (Thinking, tool_call...) continuam na paleta bege/madeira + acentos quentes
// (rust/dourado) pra erro/sucesso — mas o CONTEÚDO das mensagens do agente agora é branco;
// a mensagem do usuário continua no tom bege que já estava. O rastro de digitação
// (gradient.go) esfria em direção a um branco levemente apagado enquanto ainda está sendo
// gerado, e assenta em branco puro quando a resposta termina (renderClosedTrajectory).
var (
	colorWoodBase  = lipgloss.Color("#D8C39A") // bege/madeira claro — mensagem do usuário
	colorWoodLabel = lipgloss.Color("#C9A66B") // tan mais quente — rótulos (Thinking, tool_call...)
	colorAgentText = lipgloss.Color("#FFFFFF") // branco — corpo das mensagens do agente
	colorErrorRust = lipgloss.Color("#C97B4A") // rust — só pro RÓTULO de status de erro
	colorOKGold    = lipgloss.Color("#E0C070") // dourado quente — só pro RÓTULO de status ok
	colorBorderCol = lipgloss.Color("#5A4A35") // madeira escura — bordas
	colorStatus    = lipgloss.Color("#B8A377")

	styleThinkLabel = lipgloss.NewStyle().Foreground(colorWoodLabel).Bold(true)
	styleThinkBody  = lipgloss.NewStyle().Foreground(colorAgentText).Italic(true).PaddingLeft(2)

	styleToolCallLabel = lipgloss.NewStyle().Foreground(colorWoodLabel).Bold(true)
	styleToolCallBody  = lipgloss.NewStyle().Foreground(colorAgentText).PaddingLeft(2)

	styleToolResultOKLabel    = lipgloss.NewStyle().Foreground(colorOKGold).Bold(true)
	styleToolResultErrLabel   = lipgloss.NewStyle().Foreground(colorErrorRust).Bold(true)
	styleToolResultBody       = lipgloss.NewStyle().Foreground(colorAgentText).PaddingLeft(2)
	styleToolResultErrBodyTxt = lipgloss.NewStyle().Foreground(colorAgentText).PaddingLeft(2)

	styleFinalLabel = lipgloss.NewStyle().Foreground(colorWoodLabel).Bold(true)
	styleFinalBody  = lipgloss.NewStyle().Foreground(colorAgentText).PaddingLeft(2)

	styleUserBody = lipgloss.NewStyle().Foreground(colorWoodBase).Bold(true)

	styleViewport = lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(colorBorderCol).
			Padding(0, 1)

	styleInputBox = lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(colorWoodLabel).
			Padding(0, 1)

	styleStatusBar = lipgloss.NewStyle().
			Foreground(colorStatus).
			Padding(0, 1)

	styleTitle = lipgloss.NewStyle().
			Foreground(colorWoodLabel).
			Bold(true).
			Padding(0, 1)
)

func lipglossJoinVertical(blocks ...string) string {
	return lipgloss.JoinVertical(lipgloss.Left, blocks...)
}
