package ui

import (
	"fmt"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

type rgb [3]int

var (
	rgbGlowWhite = rgb{255, 255, 255}
	rgbWoodBase  = rgb{216, 195, 154} // #D8C39A — bege/madeira claro, cor "assentada"
)

// glowTrailLen é quantos runes finais do texto ainda em digitação recebem o gradiente —
// o resto (mais antigo) já renderiza na cor bege assentada, sem gradiente.
const glowTrailLen = 10

func hexOf(c rgb) string {
	return fmt.Sprintf("#%02X%02X%02X", c[0], c[1], c[2])
}

func lerp(a, b rgb, t float64) rgb {
	if t < 0 {
		t = 0
	}
	if t > 1 {
		t = 1
	}
	return rgb{
		int(float64(a[0]) + t*float64(b[0]-a[0])),
		int(float64(a[1]) + t*float64(b[1]-a[1])),
		int(float64(a[2]) + t*float64(b[2]-a[2])),
	}
}

// renderGlowTail renderiza `text` (o trecho ainda sendo "digitado") com um rastro de
// gradiente: os últimos caracteres (mais recentes) saem brancos e esfriam pra `base`
// conforme se afastam da ponta — efeito de destaque seguindo o cursor de geração. O trecho
// mais antigo (fora do rastro) já vai liso na cor `base`, sem gradiente, porque já "assentou".
func renderGlowTail(text string, base rgb) string {
	runes := []rune(text)
	n := len(runes)
	if n == 0 {
		return ""
	}

	flatEnd := n - glowTrailLen
	if flatEnd < 0 {
		flatEnd = 0
	}

	var b strings.Builder
	if flatEnd > 0 {
		b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(hexOf(base))).Render(string(runes[:flatEnd])))
	}
	for i := flatEnd; i < n; i++ {
		distFromNewest := n - 1 - i
		t := float64(distFromNewest) / float64(glowTrailLen)
		col := lerp(rgbGlowWhite, base, t)
		b.WriteString(lipgloss.NewStyle().Foreground(lipgloss.Color(hexOf(col))).Render(string(runes[i])))
	}
	return b.String()
}
