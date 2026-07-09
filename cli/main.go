// Comando conatus — UI do CLI harness (D13, seção M8 do PLAN.md): shell interativo em
// bubbletea conectado de verdade na API HTTP local da Ollama (modo raw, sem chat template).
// write_file/read_file/list_files executam de verdade, relativos ao diretório onde o binário
// foi chamado (os.Getwd() — não um workspace de sandbox separado). checker/shell/
// web_search ainda não estão portados para Go; chamadas pra eles recebem um erro
// UNSUPPORTED_TOOL do próprio loop (ver internal/agent), igual ao harness real faria para
// uma ferramenta desconhecida.
package main

import (
	"flag"
	"fmt"
	"os"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/devlucascfarias/Conatus-Logos/cli/internal/ollamaclient"
	"github.com/devlucascfarias/Conatus-Logos/cli/internal/ui"
)

func main() {
	model := flag.String("model", "logos-v2", "nome do modelo no Ollama")
	host := flag.String("host", "http://localhost:11434", "endereço do servidor Ollama")
	flag.Parse()

	workDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "erro obtendo diretório atual: %v\n", err)
		os.Exit(1)
	}

	client := ollamaclient.New(*model, *host)

	p := tea.NewProgram(ui.New(client, workDir), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "erro: %v\n", err)
		os.Exit(1)
	}
}
