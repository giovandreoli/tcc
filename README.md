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
- [Capturas de tela](#capturas-de-tela)
- [Métricas](#métricas)
- [Dados, privacidade e LGPD](#dados-privacidade-e-lgpd)
- [Pasta compartilhada](#pasta-compartilhada)
- [Desenvolvimento](#desenvolvimento)
- [Documentação](#documentação)
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
| **Aplicativo do paciente** | paciente, durante a sessão | `scripts\spectra.bat` |
| **Painel do fisioterapeuta** | profissional | `scripts\painel.bat` |

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

### 4. Configurar e iniciar a sessão

Em **Nova sessão**, escolha a mão tratada, o perfil de gestos, a dificuldade, a duração e os
exercícios. A opção padrão é **No navegador (local, sem comandos)**.

### 5. Sessão do paciente no navegador

1. Clique em **Iniciar sessão** no painel para gerar o código da sala.
2. Clique em **Abrir sessão do paciente**: uma nova aba recebe o código automaticamente.
3. O paciente confirma o PIN e clica em **Entrar na sessão**.
4. Clique em **Iniciar câmera** e autorize o acesso à webcam no Chrome ou Edge.
5. Os exercícios aparecem na própria imagem. Aponte e mantenha sobre os botões para avançar.
6. Clique em **Finalizar e salvar sessão**, ou pare a câmera, para salvar as métricas no histórico.

O paciente não precisa executar comandos nem instalar outro aplicativo. Para iniciar o painel,
o profissional pode dar dois cliques em `scripts\painel.bat`. O atalho procura primeiro o
ambiente `.venv-py312`, depois `.venv` e `.venv312`; as dependências devem estar instaladas nele.
Após atualizar o código, reinicie o painel para carregar a nova dependência `streamlit-webrtc`.

**Esta é a primeira etapa, local:** painel e navegador do paciente devem estar no **mesmo
computador**, acessando `http://localhost:8501`. A sala não é acessível da casa do paciente.
Não há videochamada nem áudio nesta etapa. Não publique este painel na internet.
O vídeo vai do navegador ao processo local do SPECTRA, é processado em memória e não é gravado;
o reconhecimento ainda não roda no navegador. Não há servidores STUN/TURN externos configurados.

O código de 12 caracteres expira em uma hora para entrada e exige o PIN cadastrado.
Uma sala admite um navegador por vez. Uma conexão interrompida pode ser substituída após
90 segundos sem atividade; uma sala concluída não pode ser reutilizada. Fechar a aba
normalmente encerra o vídeo e salva, mas uma queda do processo pode perder as métricas ainda
em memória: prefira sempre finalizar pelo botão e conferir o histórico.
Nesta etapa estão disponíveis abrir/fechar a mão, pinça e onda de dedos, respeitando a seleção
e a duração configuradas no painel. O início da sessão ocorre com a primeira imagem recebida.

**Próxima etapa, ainda não implementada:** hospedagem HTTPS, processamento de gestos no
navegador, acesso remoto por link/código, acompanhamento e chamada de vídeo com sinalização
e TURN, além da revisão de autenticação, consentimento e infraestrutura para uso pela internet.

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

Opções: `--hand Left|Right`, `--camera N`, `--data-dir CAMINHO`, `--no-sound`, `--verbose`.

### 6. Relatórios e evolução

Em **Histórico e evolução**: gráficos por métrica ao longo das sessões, geração e download do
PDF e do TXT, e exportação para a pasta compartilhada.

### 7. Modo demonstração

O interruptor **Modo demonstração** na barra lateral substitui o nome do paciente pelas
iniciais na tela e nos relatórios. Use-o ao projetar a apresentação do TCC.

## Capturas de tela

<!-- TODO: substituir pelos prints reais antes da entrega -->

| Tela | Imagem |
|---|---|
| Menu principal | `docs/images/menu.png` *(a incluir)* |
| Pintura livre | `docs/images/pintura.png` *(a incluir)* |
| Desenho guiado | `docs/images/desenho-guiado.png` *(a incluir)* |
| Fisioterapia | `docs/images/fisioterapia.png` *(a incluir)* |
| Painel — histórico | `docs/images/painel-historico.png` *(a incluir)* |
| Relatório em PDF | `docs/images/relatorio.png` *(a incluir)* |

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

- **Nenhum vídeo ou imagem do paciente é armazenado.** Os quadros são processados na memória e
  descartados; só os números derivados são gravados.
- Nada de paciente é versionado neste repositório. Tudo fica em um diretório de dados
  configurável, fora do repositório (padrão `~/Documents/Spectra/`, ou `%USERPROFILE%\Documents\Spectra`).
- CPF, telefone, endereço, contraindicações e anotações clínicas são **criptografados** em
  repouso; a chave fica no cofre de credenciais do sistema operacional.
- O CPF nunca aparece em nome de arquivo, log ou mensagem de erro, e é mascarado nos relatórios.
- PIN e senha usam **scrypt**, com bloqueio após 5 tentativas erradas.
- Todo acesso, exportação e exclusão fica registrado em log de auditoria.
- Exportação, anonimização e exclusão dos dados de um paciente estão no painel.

Leia [docs/lgpd.md](docs/lgpd.md) antes de coletar dados de participantes reais. Ele também
registra a pendência de verificar com o(a) orientador(a) a necessidade de aprovação por comitê
de ética.

## Pasta compartilhada

Relatórios e backups criptografados são exportados para uma pasta compartilhada configurável
(`share_dir` no arquivo de configuração). O banco de dados **permanece em disco local** e nunca
é colocado em pasta sincronizada.

> **A pasta compartilhada deve ter acesso restrito** a pessoas nomeadas — nunca "qualquer pessoa
> com o link".

Link da pasta: `<<DEFINIR: URL da pasta compartilhada com acesso restrito>>`

## Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

black . && isort . && ruff check . && pytest
pytest --cov --cov-report=term-missing   # cobertura
```

A suíte roda **headless**: sem câmera, sem janela e sem `winsound`. As mãos usadas nos testes
são sintéticas (`tests/conftest.py`).

Estrutura:

```
spectra/
  app.py  config.py  service.py  consent.py  __main__.py
  i18n/            carregador de catálogo
  locales/         pt_BR.json, tcle_pt_BR.md
  detection/       câmera, MediaPipe, constantes de landmarks
  gestures/        ângulos, calibração, perfis, máquina de estados
  core/            canvas, rastro, suavização, limites de sessão
  ui/              botões, widgets, teclado por gestos, som
  modes/           menu, pintura, desenho guiado, educativos, fisioterapia
  guided/          figuras-alvo e pontuação
  metrics/         ADM, tremor, fadiga, precisão, qualidade, gravador
  storage/         SQLite, criptografia, auditoria, backup
  reports/         objeto de relatório, PDF, TXT, gráficos
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
| [docs/lgpd.md](docs/lgpd.md) | proteção de dados, direitos do titular e ética em pesquisa |
| [docs/decisions.md](docs/decisions.md) | registro de todas as decisões técnicas e seus motivos |

## Licença

MIT. Veja `pyproject.toml`.
