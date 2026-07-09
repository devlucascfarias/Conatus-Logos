package ui

import "github.com/charmbracelet/lipgloss"

var (
	colorThink      = lipgloss.Color("245") // cinza — raciocínio interno
	colorToolCall   = lipgloss.Color("39")  // azul — ação
	colorToolOK     = lipgloss.Color("42")  // verde — resultado de ferramenta ok
	colorToolError  = lipgloss.Color("203") // vermelho — resultado de ferramenta com erro
	colorFinal      = lipgloss.Color("255") // branco — resposta pública
	colorUser       = lipgloss.Color("213") // rosa — mensagem do usuário
	colorMuted      = lipgloss.Color("240")
	colorAccent     = lipgloss.Color("212")
	colorBorderCol  = lipgloss.Color("237")
	colorStatusText = lipgloss.Color("250")

	styleThinkLabel = lipgloss.NewStyle().Foreground(colorThink).Italic(true)
	styleThinkBody  = lipgloss.NewStyle().Foreground(colorThink).Italic(true).PaddingLeft(2)

	styleToolCallLabel = lipgloss.NewStyle().Foreground(colorToolCall).Bold(true)
	styleToolCallBody  = lipgloss.NewStyle().Foreground(colorToolCall).PaddingLeft(2)

	styleToolResultOKLabel    = lipgloss.NewStyle().Foreground(colorToolOK).Bold(true)
	styleToolResultErrLabel   = lipgloss.NewStyle().Foreground(colorToolError).Bold(true)
	styleToolResultBody       = lipgloss.NewStyle().Foreground(colorMuted).PaddingLeft(2)
	styleToolResultErrBodyTxt = lipgloss.NewStyle().Foreground(colorToolError).PaddingLeft(2)

	styleFinalLabel = lipgloss.NewStyle().Foreground(colorAccent).Bold(true)
	styleFinalBody  = lipgloss.NewStyle().Foreground(colorFinal).PaddingLeft(2)

	styleUserLabel = lipgloss.NewStyle().Foreground(colorUser).Bold(true)
	styleUserBody  = lipgloss.NewStyle().Foreground(colorUser)

	styleViewport = lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(colorBorderCol).
			Padding(0, 1)

	styleInputBox = lipgloss.NewStyle().
			BorderStyle(lipgloss.RoundedBorder()).
			BorderForeground(colorAccent).
			Padding(0, 1)

	styleStatusBar = lipgloss.NewStyle().
			Foreground(colorStatusText).
			Padding(0, 1)

	styleTitle = lipgloss.NewStyle().
			Foreground(colorAccent).
			Bold(true).
			Padding(0, 1)
)

func lipglossJoinVertical(blocks ...string) string {
	return lipgloss.JoinVertical(lipgloss.Left, blocks...)
}
