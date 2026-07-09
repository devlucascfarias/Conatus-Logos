package ui

import "github.com/charmbracelet/lipgloss"

// Paleta bege/madeira — um tom só, variando peso (label bold vs corpo normal) e um acento
// quente (rust/dourado) só pra erro/sucesso de ferramenta, onde a distinção importa de
// verdade. O rastro de digitação (gradient.go) sempre esfria em direção a colorWoodBase.
var (
	colorWoodBase  = lipgloss.Color("#D8C39A") // bege/madeira claro — texto assentado
	colorWoodLabel = lipgloss.Color("#C9A66B") // tan mais quente — rótulos (Thinking, tool_call...)
	colorWoodMuted = lipgloss.Color("#A9895F") // marrom mais escuro — corpo secundário (tool_result)
	colorErrorRust = lipgloss.Color("#C97B4A") // rust — só pra status de erro
	colorOKGold    = lipgloss.Color("#E0C070") // dourado quente — só pra status ok
	colorBorderCol = lipgloss.Color("#5A4A35") // madeira escura — bordas
	colorStatus    = lipgloss.Color("#B8A377")

	styleThinkLabel = lipgloss.NewStyle().Foreground(colorWoodLabel).Bold(true)
	styleThinkBody  = lipgloss.NewStyle().Foreground(colorWoodBase).Italic(true).PaddingLeft(2)

	styleToolCallLabel = lipgloss.NewStyle().Foreground(colorWoodLabel).Bold(true)
	styleToolCallBody  = lipgloss.NewStyle().Foreground(colorWoodMuted).PaddingLeft(2)

	styleToolResultOKLabel    = lipgloss.NewStyle().Foreground(colorOKGold).Bold(true)
	styleToolResultErrLabel   = lipgloss.NewStyle().Foreground(colorErrorRust).Bold(true)
	styleToolResultBody       = lipgloss.NewStyle().Foreground(colorWoodMuted).PaddingLeft(2)
	styleToolResultErrBodyTxt = lipgloss.NewStyle().Foreground(colorErrorRust).PaddingLeft(2)

	styleFinalLabel = lipgloss.NewStyle().Foreground(colorWoodLabel).Bold(true)
	styleFinalBody  = lipgloss.NewStyle().Foreground(colorWoodBase).PaddingLeft(2)

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
