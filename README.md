# SPECTRA

**S**istema de **P**intura e **E**ducação **C**om **T**ecnologia de **R**econhecimento de m**A**os

Plataforma de reabilitação de mão, dedos e punho controlada **apenas por gestos** captados por
webcam. Durante a sessão o paciente não usa teclado nem mouse: pinta livremente, executa
desafios de desenho guiado e exercícios de fisioterapia. O fisioterapeuta acompanha a evolução
por um painel e por relatórios em PDF e TXT.

> **Trabalho de Conclusão de Curso. Protótipo de pesquisa — não é um dispositivo médico
> certificado.** As medidas angulares vêm de uma webcam 2D, são **estimativas** e **não
> substituem goniometria** nem avaliação profissional.

---

## Sumário

- [Como funciona](#como-funciona)
- [Gestos](#gestos)
- [Requisitos](#requisitos)
- [Instalação no Windows](#instalação-no-windows)
- [Como usar](#como-usar)
  - [Exercícios sob medida](#4-montar-exercícios-sob-medida-opcional)
  - [Laudos e documentos](#5-laudos-e-documentos)
  - [Sessão pela internet](#sessão-pela-internet-paciente-em-casa)
- [Aprendizado e pesquisa](#aprendizado-e-pesquisa)
- [Capturas de tela](#capturas-de-tela)
- [Métricas](#métricas)
- [Dados, privacidade e LGPD](#dados-privacidade-e-lgpd)
- [Pasta compartilhada](#pasta-compartilhada)
- [Desenvolvimento](#desenvolvimento)
- [Documentação](#documentação)
- [Hospedagem gratuita](#hospedagem-gratuita)
- [Licença](#licença)

---

## Como funciona

```
webcam ──► MediaPipe ──► ângulos das ──► máquina de ──► modo ativo ──► métricas ──► relatório
            (21 pontos)   articulações     estados         (pintura,      (ADM,        (PDF/TXT)
                                           com histerese    guiado,        tremor,
                                           e tempo de       fisioterapia)  fadiga)
                                           confirmação
```

Duas aplicações:

| Aplicação | Para quem | Como abrir |
|---|---|---|
| **Painel do fisioterapeuta** | profissional: pacientes, TCLE, exercícios, laudos, salas e relatórios | `scripts\painel.bat` |
| **Página do paciente** | paciente, no navegador, no mesmo computador ou em casa | link gerado pelo painel |
| Aplicativo Windows (anterior) | paciente, com pintura e modos educativos | `scripts\spectra.bat` |

## Gestos

Perfil **padrão** (configurável por paciente no painel):

| Gesto | Ação |
|---|---|
| 1 dedo (indicador) | apontar e navegar |
| 2 dedos (indicador + médio) | pintar |
| 3 dedos (indicador + médio + anelar) | apagar |
| mão aberta | abrir o menu de cores |
| punho fechado | pausar / retomar |

Dentro do menu, a cor é escolhida **apontando e mantendo** (cerca de 1,2 s) sobre a amostra. Os
botões da tela funcionam do mesmo jeito.

Gestos adicionais, em todos os modos do aplicativo:

| Gesto | Ação |
|---|---|
| **pinça** (encostar polegar e indicador) | **clique** imediato no botão sob o cursor |
| polegar no **médio / anelar / mínimo** | escolhe **vermelho / verde / azul** no desenho (configurável) |
| **aproximar / afastar** a mão da câmera | traço mais grosso / mais fino no desenho; zoom no modo ambiente estéril |

O cursor é suavizado por um **Filtro de Kalman**. A profundidade vem da área do polígono da
palma, corrigida pela razão de aspecto, para que **girar a mão não seja confundido com
aproximar**. Detalhes e resultados em [docs/learning.md](docs/learning.md).

Existe também o perfil **simplificado**, para pouca destreza: pintar é fazer uma **pinça**
(polegar encostando no indicador), o que dispensa controle independente dos dedos. Perfis
personalizados podem usar qualquer uma das 32 combinações; o painel avisa sobre combinações
ambíguas ou anatomicamente difíceis.

Detalhes e limites em [docs/gestures.md](docs/gestures.md).

### Segurança durante a sessão

- limite de tempo (padrão 20 minutos, configurável);
- aviso de descanso a cada 5 minutos;
- gesto de pausa fácil (punho fechado);
- `ESC` encerra imediatamente — é a saída de emergência do fisioterapeuta.

**Interrompa a sessão se o paciente sentir dor.**

## Requisitos

- **Windows 10 ou 11** com webcam (aplicativo do paciente)
- **Python 3.10 a 3.12** — o MediaPipe ainda não publica *wheels* para 3.13 no Windows
- Iluminação razoável e a mão inteira dentro do enquadramento

Linux/WSL é suportado **apenas** para desenvolvimento e testes (sem câmera, sem janelas).

## Instalação no Windows

```powershell
git clone <url-do-repositorio>
cd tcc

py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Instale as dependências em conjunto pelo arquivo de requisitos. O MediaPipe 0.10.21
exige `protobuf>=4.25.3,<5`; por isso o painel usa Streamlit 1.50.0 e pandas 2.3.3.
Atualizar o Streamlit isoladamente pode tornar o ambiente incompatível.
Confira a instalação com `python -m pip check`.

O modelo `hand_landmarker.task` já acompanha o repositório; se faltar, é baixado
automaticamente na primeira execução.

## Como usar

### 1. Primeiro acesso ao painel

```powershell
scripts\painel.bat
```

Crie a conta de fisioterapeuta (senha de 8 caracteres ou mais) e faça login.

### 2. Cadastrar o paciente

Em **Pacientes**, preencha os dados e defina um **PIN de 4 dígitos**. O CPF é validado pelos
dígitos verificadores.

### 3. Registrar o TCLE

Em **Consentimento (TCLE)**, leia o termo com o participante e registre o aceite.

> **Nenhuma sessão pode ser iniciada sem consentimento registrado.** Isso é uma regra do banco
> de dados, não apenas da interface.

### 4. Montar exercícios sob medida (opcional)

Em **Exercícios do paciente**, o fisioterapeuta cria exercícios a partir de **movimentos que o
SPECTRA já reconhece de forma validada** e ajusta os parâmetros dentro de limites seguros:

| Movimento base | Parâmetros ajustáveis |
|---|---|
| Abrir e fechar a mão | repetições, tempo segurando o punho fechado |
| Pinça (polegar com um dedo) | dedo, distância do toque (ampla/padrão/precisa), tempo segurando |
| Onda de dedos | ordem (indicador → mínimo ou o inverso) |
| Levantar um dedo | dedo, exigir os outros dobrados (isolado), tempo segurando |
| Oposição do polegar | tocar indicador, médio, anelar e mínimo em sequência; distância do toque |

Nome e instruções são livres; se as instruções ficarem vazias, o SPECTRA as gera a partir do
movimento. Valores fora dos limites (por exemplo, 0 ou 31 repetições, mais de 5 s segurando)
são recusados. Use **Testar um exercício com a câmera** para fazer o movimento e conferir a
contagem antes de prescrever. Arquivar um exercício o retira de novas sessões sem alterar o
histórico das anteriores.

### 5. Laudos e documentos

Em **Laudos e documentos**, cada paciente tem seu prontuário de documentos:

- **Escrever laudo:** laudo fisioterapêutico, evolução ou encaminhamento em texto, que pode
  ser baixado em PDF para impressão;
- **Anexar arquivo:** laudo médico, exame ou encaminhamento em **PDF, PNG ou JPG** (até 10 MB).
  O tipo é conferido pelo conteúdo do arquivo, não pela extensão.

Tudo é **criptografado** neste computador, entra no backup criptografado, aparece na
exportação LGPD e é apagado na exclusão ou anonimização do paciente. Abrir, baixar ou excluir
um documento fica registrado na auditoria.

### 6. Configurar e iniciar a sessão

Em **Nova sessão**, escolha a mão tratada, o perfil de gestos, a dificuldade, a duração e os
exercícios (básicos e personalizados do paciente). Clique em **Iniciar sessão** para gerar o
código da sala, válido por 1 hora e protegido pelo PIN do paciente.

- **Paciente neste computador:** clique em **Abrir sessão do paciente**.
- **Paciente em casa:** veja [Sessão pela internet](#sessão-pela-internet-paciente-em-casa).

### 7. Sessão do paciente no navegador

1. **Entrar:** o paciente abre o link (ou digita o código), confirma o PIN e clica em
   **Entrar na sessão**.
2. **Preparar:** informa a dor atual (escala 0–10), confirma a lista de preparo (mão visível,
   boa iluminação, distância, parar se doer) e vê os exercícios do dia com a meta de cada um.
   Com dor **8 ou mais**, o exercício é bloqueado e o paciente é orientado a falar com o
   fisioterapeuta.
3. **Exercitar:** clique em **Iniciar câmera** e autorize a webcam (Chrome, Edge ou Firefox).
   Na primeira vez o navegador carrega o modelo de reconhecimento da mão (cerca de 20 MB).
   Ao lado do vídeo ficam exercício, repetições, tempo restante, a barra de "segure a posição"
   e as dicas do Spectro, que dizem o próximo passo ("Toque o polegar no anelar", "Agora
   solte"). Use **Próximo exercício** para avançar.
   **Esquerda/direita se referem à sua mão, não ao lado da tela.** Lembretes de descanso
   aparecem ao lado do vídeo.
4. Clique em **Finalizar e salvar sessão**, ou pare a câmera, para salvar as métricas.
5. **Concluir:** o paciente vê o resumo e informa a dor após a sessão. Dor 8 ou mais, ou
   aumento de 3 pontos ou mais, exibe um aviso para conversar com o fisioterapeuta.

Enquanto isso, a tela **Início** do painel mostra as salas ao vivo (aguardando, em andamento,
concluída, interrompida), com exercício atual, repetições, mão visível e progresso.

**Como o vídeo é tratado:** o reconhecimento da mão (MediaPipe) roda **no navegador do
paciente**, com arquivos servidos pelo próprio SPECTRA, sem CDN. Apenas 21 pontos da mão por
quadro são enviados ao painel; **nenhuma imagem ou vídeo sai do computador do paciente** e
nada é gravado. Isso também dispensa servidores de vídeo (STUN/TURN) e funciona em conexões
domésticas comuns.

Uma sala admite um navegador por vez. Uma conexão interrompida pode ser retomada após
90 segundos sem atividade; uma sala concluída não pode ser reutilizada. Fechar a aba encerra
e salva, mas uma queda do painel pode perder métricas ainda em memória: prefira finalizar
pelo botão e conferir o histórico.

### Sessão pela internet (paciente em casa)

O SPECTRA gera um **endereço https gratuito** com o
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
(modo *quick tunnel*: sem conta, sem cartão, sem configurar roteador):

1. Em **Nova sessão**, no quadro **Acesso pela internet**, clique em **Baixar cloudflared**
   na primeira vez (o programa oficial vem do GitHub da Cloudflare; também pode ser instalado
   com `winget install Cloudflare.cloudflared`).
2. Clique em **Ativar link pela internet**. Em alguns segundos aparece um endereço como
   `https://palavras-aleatorias.trycloudflare.com`.
3. Inicie a sessão: o painel mostra o **link do paciente**, que já abre na sala certa. Envie
   o link por um canal privado; o paciente ainda precisa do **PIN**.
4. Ao terminar, clique em **Desativar link**.

Segurança do acesso remoto:

- Pelo endereço público **só abre a página do paciente** (código da sala + PIN, com bloqueio
  após 5 tentativas erradas). O painel do fisioterapeuta, os prontuários e os laudos
  respondem **apenas** a quem está neste computador; o SPECTRA reconhece o acesso externo pelo
  endereço e pela origem da página, e nega o painel mesmo com uma sessão de login copiada.
- O painel continua escutando só em `localhost`; quem se conecta à internet é o cloudflared.
- O link vale enquanto o painel e este computador estiverem ligados e muda a cada ativação.
- O *quick tunnel* da Cloudflare é gratuito e destinado a testes e demonstrações, sem garantia
  de disponibilidade. Para uso clínico contínuo, avalie um túnel com conta e domínio próprios.

#### Aplicativo Windows anterior (opcional)

Selecione **Aplicativo Windows (modo anterior)** no painel para obter o identificador e o comando:

```powershell
scripts\spectra.bat --session <id-da-sessao>
```

O paciente confirma o **PIN no teclado numérico por gestos** e a sessão começa. Ao sair, as
métricas são calculadas e gravadas automaticamente.

Para uso livre, sem gravar nada:

```powershell
scripts\spectra.bat
```

Opções: `--hand Left|Right`, `--camera N`, `--data-dir CAMINHO`, `--filter kalman|exponential|raw`,
`--usability CÓDIGO`, `--no-sound`, `--verbose`.

O menu do aplicativo inclui o modo **Ambiente estéril**: um visualizador de imagens operado sem
toque, pensado para centros cirúrgicos. Dois dedos avançam, três voltam, aproximar a mão amplia,
afastar reduz, mão aberta restaura o zoom e **punho fechado trava/destrava** todos os comandos.
As imagens são sintéticas; nenhum exame real é usado.

## Aprendizado e pesquisa

Seção **Aprendizado e pesquisa** do painel, com cinco abas:

| Aba | O que faz |
|---|---|
| 🎓 **Trilha de gestos** | 8 lições curtas, conferidas pela câmera, ensinam a interface (mostrar a mão, abrir, fechar, apontar, clique por pinça, cores nos dedos, aproximar, girar sem aproximar). Mede o tempo de aprendizagem e dá estrelas. O paciente também pode fazê-la antes dos exercícios. |
| 🧠 **Gestos personalizados (IA)** | O fisioterapeuta grava alguns segundos de cada gesto do paciente, treina um classificador k-NN (sem GPU), vê a acurácia por validação cruzada e a matriz de confusão, testa ao vivo e associa cada gesto a um comando. O modelo vai para o prontuário e para o aplicativo. |
| 🤏 **Calibrar clique** | Aprende os limites de pinça da mão do paciente, para quem tem pouca oposição do polegar. |
| 🧪 **Testes com usuários** | Participantes anônimos (P01…), comandos da tarefa de apontar (ISO 9241-9) com filtros em ordem contrabalanceada, questionário **SUS**, comparação estatística (teste t pareado) e exportação em **CSV**. |
| 📊 **Avaliação dos algoritmos** | Compara Kalman × média exponencial × sem filtro e os três estimadores de profundidade em dados sintéticos de verdade conhecida. |

As mesmas tabelas saem no terminal, para o capítulo de resultados:

```powershell
.venv-py312\Scripts\python.exe -m spectra.evaluation
```

Resultados atuais (detalhes e método em [docs/learning.md](docs/learning.md)):

- **Kalman:** reduz o tremor do cursor em cerca de 73% e tem o menor erro em traços em
  movimento (4,0 px, contra 5,9 px sem filtro e 14,7 px da média exponencial, que atrasa).
- **Profundidade:** só a área erra até 52% quando a mão gira. A razão de aspecto mantém o
  erro médio em cerca de 2% em todos os cenários. A distância pulso–dedo médio corrige a
  pronação, mas não a flexão do punho.

Protocolo de teste por participante (cerca de 20 minutos): TCLE → trilha de gestos → tarefa de
apontar com os dois filtros → SUS. Os comandos aparecem prontos no painel, por exemplo:

```powershell
scripts\spectra.bat --usability P01 --filter kalman
scripts\spectra.bat --usability P01 --filter exponential
```

### 8. Relatórios e evolução

Em **Histórico e evolução**: gráficos por métrica ao longo das sessões, dor antes e depois de
cada sessão, geração e download do PDF e do TXT, e exportação para a pasta compartilhada.

A tela **Início** resume a semana: pacientes, sessões e minutos em 7 dias, alívio médio de dor
e pacientes sem TCLE ativo, com dicas do Spectro sobre o próximo passo.

### Interface e acessibilidade

- Animação de abertura com o mascote **Spectro** (uma vez por sessão do navegador).
- Painel **Acessibilidade** com **Texto maior** e **Alto contraste**, no painel e na página
  do paciente. As animações respeitam a preferência de movimento reduzido do sistema.
- Tudo funciona offline: sem fontes, scripts ou imagens externas.

### 9. Modo demonstração

O interruptor **Modo demonstração** na barra lateral substitui o nome do paciente pelas
iniciais na tela e nos relatórios. Use-o ao projetar a apresentação do TCC.

## Capturas de tela

As capturas reais devem ser adicionadas antes da entrega final. Use dados fictícios e não inclua
rosto, nome, CPF, diagnóstico, relatórios ou qualquer dado de participante.

## Métricas

Calculadas **sempre a partir dos pontos da mão, nunca de vídeo**:

- **Amplitude de movimento** por dedo (mínimo, máximo e amplitude, em graus)
- **Repetições e ritmo** por exercício
- **Precisão do traçado** no desenho guiado (desvio médio e máximo, percurso, tempo, pontuação)
- **Qualidade do rastreamento** (percentual de quadros com mão, confiança média)
- **Suavidade e tremor** (jerk e energia espectral na faixa de 4–12 Hz)
- **Fadiga** (queda de amplitude e de ritmo ao longo da sessão)
- **Oposição do polegar** e **estimativa de punho**

Toda medida derivada de ângulo 2D aparece marcada como **estimativa**. A definição e os limites
de cada métrica estão em [docs/metrics.md](docs/metrics.md) — vale a leitura antes de
interpretar qualquer número.

## Dados, privacidade e LGPD

- **Nenhum vídeo ou imagem do paciente é armazenado.** Na sessão pelo navegador, o
  reconhecimento roda no computador do paciente e só os 21 pontos da mão chegam ao painel; só
  os números derivados são gravados.
- Nada de paciente é versionado neste repositório. Tudo fica em um diretório de dados
  configurável, fora do repositório (padrão `~/Documents/Spectra/`, ou `%USERPROFILE%\Documents\Spectra`).
- CPF, telefone, endereço, contraindicações, anotações clínicas, **laudos e anexos** são
  **criptografados** em repouso; a chave fica no cofre de credenciais do sistema operacional.
- O CPF nunca aparece em nome de arquivo, log ou mensagem de erro, e é mascarado nos relatórios.
- PIN e senha usam **scrypt**, com bloqueio após 5 tentativas erradas.
- Todo acesso, exportação e exclusão fica registrado em log de auditoria.
- Exportação, anonimização e exclusão dos dados de um paciente estão no painel.
- Pela internet, apenas a página do paciente fica acessível; o painel responde só localmente.

Leia [docs/lgpd.md](docs/lgpd.md) antes de coletar dados de participantes reais. Ele também
registra a pendência de verificar com o(a) orientador(a) a necessidade de aprovação por comitê
de ética.

## Pasta compartilhada

Relatórios e backups criptografados são exportados para uma pasta compartilhada configurável
(`share_dir` no arquivo de configuração). O banco de dados **permanece em disco local** e nunca
é colocado em pasta sincronizada.

> **A pasta compartilhada deve ter acesso restrito** a pessoas nomeadas — nunca "qualquer pessoa
> com o link".

Não há link de compartilhamento definido no repositório. Configure `share_dir` no arquivo local
`config.json` e não publique links, relatórios, backups ou dados de participantes no GitHub Pages.

## Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-docs.txt

python -m black --check .
python -X utf8 -m isort --check-only spectra tests
python -m ruff check .
python -m pytest
python -m pytest --cov --cov-report=term-missing   # cobertura
```

A suíte roda **headless**: sem câmera, sem janela e sem `winsound`. As mãos usadas nos testes
são sintéticas (`tests/conftest.py`).

O build de documentação executado pelo CI usa `python -m mkdocs build --strict`.

Estrutura:

```
spectra/
  app.py  config.py  service.py  consent.py  __main__.py
  i18n/            carregador de catálogo
  locales/         pt_BR.json, tcle_pt_BR.md
  detection/       câmera, MediaPipe, constantes de landmarks
  gestures/        ângulos, calibração, perfis, máquina de estados
  core/            canvas, rastro, Filtro de Kalman, limites de sessão
  ui/              botões, widgets, teclado por gestos, som
  modes/           menu, pintura, desenho guiado, educativos, fisioterapia, ambiente estéril, teste
  exercises/       exercícios sob medida: movimentos validados, detectores, executor
  learning/        classificador de gestos (k-NN) e trilha de gestos
  evaluation/      mão 3D sintética e avaliação reprodutível dos algoritmos
  domains.py       gestos → comandos por domínio (reabilitação, estéril, desenho)
  usability.py     SUS, tarefa de apontar (Fitts/ISO 9241-9), estatística
  guided/          figuras-alvo e pontuação
  metrics/         ADM, tremor, fadiga, precisão, qualidade, gravador
  storage/         SQLite, criptografia, auditoria, backup, laudos
  reports/         objeto de relatório, PDF, TXT, gráficos, laudo em PDF
  browser/         página do paciente, salas, câmera com MediaPipe no navegador
  remote.py        link gratuito pela internet (Cloudflare Tunnel) e controle de acesso
  therapist_panel/ painel Streamlit
tests/unit/        testes de lógica pura
docs/              decisões, arquitetura, gestos, métricas, LGPD
scripts/           atalhos .bat para Windows
```

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/architecture.md](docs/architecture.md) | camadas, fluxo de dados e por que o código é separado assim |
| [docs/gestures.md](docs/gestures.md) | motor de gestos, calibração, perfis e validação |
| [docs/metrics.md](docs/metrics.md) | definição, interpretação e **limites** de cada métrica |
| [docs/learning.md](docs/learning.md) | Kalman, profundidade Z, clique, IA de gestos, trilha, testes com usuários e resultados |
| [docs/lgpd.md](docs/lgpd.md) | proteção de dados, direitos do titular e ética em pesquisa |
| [docs/decisions.md](docs/decisions.md) | registro de todas as decisões técnicas e seus motivos |

## Hospedagem gratuita

Tudo no SPECTRA funciona sem custo:

| O quê | Onde | Custo |
|---|---|---|
| Painel, banco de dados, laudos | computador do fisioterapeuta (`localhost`) | grátis |
| Acesso do paciente em casa | [Cloudflare quick tunnel](#sessão-pela-internet-paciente-em-casa) (`*.trycloudflare.com`) | grátis, sem conta |
| Reconhecimento da mão | navegador do paciente (MediaPipe, arquivos servidos pelo painel) | grátis |
| Documentação do TCC | GitHub Pages | grátis em repositório público |
| Testes automáticos (Windows e Linux, Python 3.10–3.12) | GitHub Actions | grátis em repositório público |

O banco **não** vai para a nuvem: dados de saúde ficam criptografados no computador do
fisioterapeuta, e a internet só alcança a página do paciente.

**Documentação no GitHub Pages.** O workflow constrói o site com MkDocs, trata avisos como
erros e publica as alterações feitas na branch `main`. Para ativar:

1. No GitHub, abra **Settings → Pages** e selecione **GitHub Actions** em *Build and deployment*.
2. Faça merge das alterações na branch `main`; acompanhe o workflow **Publish documentation**
   em **Actions**.
3. O site ficará disponível em `https://giovandreoli.github.io/tcc/` após a primeira publicação.

Esse site contém **somente documentação pública**. Não publique dados de pacientes,
relatórios, backups, chaves ou arquivos de configuração, e não torne o repositório público se
houver qualquer dado restrito nele.

## Licença

MIT, veja [LICENSE](LICENSE). O MediaPipe Tasks Vision distribuído em
`spectra/browser/camera/vendor/` é Apache-2.0 (licença na mesma pasta).
