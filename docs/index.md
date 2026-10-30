# SPECTRA

Sistema de pintura e educação com tecnologia de reconhecimento de mãos para apoiar exercícios
de reabilitação de mão, dedos e punho. O projeto é um protótipo acadêmico e não é um dispositivo
médico certificado.

## Comece por aqui

- [Como instalar e executar o aplicativo](https://github.com/giovandreoli/tcc#instalação-no-windows)
- [Arquitetura e fluxo de dados](architecture.md)
- [Gestos e calibração](gestures.md)
- [Métricas e limitações](metrics.md)
- [Aprendizado, Kalman, profundidade e testes com usuários](learning.md)
- [Privacidade, LGPD e ética](lgpd.md)
- [Decisões técnicas](decisions.md)

## Segurança e escopo

O painel do fisioterapeuta e os prontuários ficam **no computador do fisioterapeuta**. O
paciente pode fazer a sessão no mesmo computador ou de casa, por um link https gratuito; nesse
caso o reconhecimento da mão roda **no navegador do paciente** e apenas 21 pontos da mão por
quadro são enviados, nunca vídeo. Pela internet só abre a página do paciente, protegida por
código da sala e PIN. Este site é documentação estática: não hospeda o aplicativo e não recebe
dados de pacientes.

As medidas derivadas de uma webcam 2D são estimativas de pesquisa. Elas não substituem
goniometria, avaliação profissional ou orientação clínica.
