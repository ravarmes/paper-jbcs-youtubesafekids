# Annotation guideline — toxicity in transcribed video sentences

This is the guideline used to label the corpus in `data/corpus_toxicidade_publico.csv`.
The three level definitions are the ones stated in Section 4.3 of the paper; the decision
procedure and the contrastive examples make explicit the criterion that separates Mild
from Severe — the **target** of the offence, not the strength of the word.

The Portuguese text is the original given to the raters. The English translation is a
reading aid. The examples "porcaria de carro" and "Este idiota do Robson fez isso" are
the ones in the paper; "Que porra de ônibus" is the one in the prompt of the generative
baseline (`app/nlp/training/train_toxicity_llm_baselines.py`); the other examples were
written for the guideline.

Each rater worked alone and saw only the sentence: no neighbouring sentences, no
transcript, no video, no model prediction and no previous label. The final label is the
mode of the three annotations; when the three differ, the label of rater P1 (the
professor) is used.

---

## Original (Português)

### A tarefa

Você vai ler frases transcritas da fala de vídeos do YouTube e classificar cada uma em um
de três níveis de toxicidade: **Nenhuma**, **Leve** ou **Severa**. O critério segue a
distinção entre agressão direta e indireta usada na literatura da área.

### Como trabalhar

- Trabalhe sozinho. Não consulte os outros anotadores nem compare respostas antes de entregar.
- Julgue só a frase, como está escrita. Não abra o vídeo, não procure o contexto e não
  imagine quem fala nem em que tom.
- A frase vem de transcrição automática e pode ter palavras trocadas ou erros de
  digitação. Julgue pelo sentido evidente.
- Preencha todas as linhas usando a lista da coluna `rotulo`. Na dúvida, escolha o nível
  que a definição descreve melhor e anote a dúvida na coluna `observacao`.

### Os três níveis

- **Nenhuma:** texto apropriado, sem linguagem agressiva.
- **Leve:** ofensa indireta, ou seja, crítica dirigida a objetos, situações ou entidades
  abstratas, **sem alvo pessoal**. Ex.: "porcaria de carro"; "Que porra de ônibus".
- **Severa:** ofensa direta, ou seja, insulto, ameaça ou discurso de ódio **dirigido
  explicitamente a uma pessoa ou grupo**. Ex.: "Este idiota do Robson fez isso".

### Como decidir: duas perguntas, nesta ordem

1. A frase tem linguagem agressiva ou ofensiva (xingamento, desprezo, hostilidade, ameaça)?
   Se não tem, é **Nenhuma**.
2. A ofensa atinge uma pessoa ou um grupo?
   - Se sim, é **Severa**.
   - Se atinge uma coisa, situação, lugar ou ideia, é **Leve**.

### O que mais pesa na decisão

**1. O que separa Leve de Severa é o alvo, não a força da palavra.** Um palavrão pesado
contra uma coisa é Leve. Um xingamento fraco contra uma pessoa é Severa.

| Leve (alvo: coisa, situação, ideia) | Severa (alvo: pessoa ou grupo) |
|---|---|
| Que filme idiota. | Você é idiota. |
| Essa porcaria não funciona. | Você é uma porcaria. |
| Que regra ridícula. | Que professor ridículo. |
| Que merda de dia. | Cala a boca, seu bobo. |

**2. A pessoa pode aparecer de vários jeitos:** em segunda pessoa ("você", "seu
idiota"), em terceira ("aquele cara", "o Robson"), por nome, apelido ou vocativo.
Personagens de desenho contam como pessoas.

**3. Grupos também são alvo.** Ofensa contra pessoas por raça, religião, gênero,
orientação sexual, origem ou deficiência é **Severa** (discurso de ódio).

**4. Ameaça a alguém é Severa**, mesmo sem palavrão: "Eu vou te pegar lá fora."

**5. Palavrão não decide sozinho.**
- Como intensificador, elogio ou espanto, sem ofender ninguém, é **Nenhuma**: "Essa música é foda!"
- Para reclamar de uma coisa ou situação, é **Leve**: "Que porra de ônibus."
- Contra alguém, é **Severa**: "Vai se foder, seu otário."
- E uma frase sem nenhum palavrão pode ser Severa: "Você é um inútil, ninguém te suporta."

**6. Criticar o que alguém fez, sem insultar a pessoa, não é Severa.**
- Com tom ofensivo, é **Leve**: "Que ideia ridícula."
- Sem tom ofensivo, é **Nenhuma**: "Acho que você errou essa conta."

---

## English translation

### The task

You will read sentences transcribed from the speech of YouTube videos and assign each one
to one of three toxicity levels: **None** (*Nenhuma*), **Mild** (*Leve*) or **Severe**
(*Severa*). The criterion follows the distinction between direct and indirect aggression
used in the literature.

### How to work

- Work alone. Do not consult the other raters or compare answers before handing in.
- Judge only the sentence, as written. Do not open the video, do not look for context and
  do not imagine who is speaking or in what tone.
- The sentence comes from automatic transcription and may contain wrong words or typos.
  Judge by the evident meaning.
- Fill every row using the list in the `rotulo` column. When in doubt, choose the level the
  definition describes best and note the doubt in the `observacao` column.

### The three levels

- **None:** appropriate text, free of aggressive language.
- **Mild:** indirect offensiveness, that is, criticism directed at objects, situations or
  abstract entities, **with no personal target**. E.g. "porcaria de carro" ('piece of junk
  car'); "Que porra de ônibus" ('what a fucking bus').
- **Severe:** direct offensiveness, that is, insult, threat or hate speech **explicitly
  directed at a person or group**. E.g. "Este idiota do Robson fez isso" ('this idiot
  Robson did this').

### How to decide: two questions, in this order

1. Does the sentence contain aggressive or offensive language (swearing, contempt,
   hostility, threat)? If not, it is **None**.
2. Does the offence reach a person or a group?
   - If so, it is **Severe**.
   - If it reaches a thing, situation, place or idea, it is **Mild**.

### What weighs most

**1. What separates Mild from Severe is the target, not the strength of the word.** A
strong swear word against a thing is Mild. A weak insult against a person is Severe.

| Mild (target: thing, situation, idea) | Severe (target: person or group) |
|---|---|
| Que filme idiota. ('What a stupid film.') | Você é idiota. ('You are stupid.') |
| Essa porcaria não funciona. ('This junk does not work.') | Você é uma porcaria. ('You are junk.') |
| Que regra ridícula. ('What a ridiculous rule.') | Que professor ridículo. ('What a ridiculous teacher.') |
| Que merda de dia. ('What a shitty day.') | Cala a boca, seu bobo. ('Shut up, you fool.') |

**2. The person can appear in several ways:** in the second person ("você", "seu
idiota"), in the third ("aquele cara", "o Robson"), by name, nickname or vocative.
Cartoon characters count as people.

**3. Groups are targets too.** Offence against people because of race, religion, gender,
sexual orientation, origin or disability is **Severe** (hate speech).

**4. A threat against someone is Severe**, even without swearing: "Eu vou te pegar lá
fora." ('I'll get you outside.')

**5. Swearing does not decide on its own.**
- As an intensifier, praise or surprise, offending no one, it is **None**: "Essa música é
  foda!" ('This song is fucking great!')
- To complain about a thing or situation, it is **Mild**: "Que porra de ônibus."
- Against someone, it is **Severe**: "Vai se foder, seu otário." ('Fuck off, you sucker.')
- A sentence with no swear word at all can be Severe: "Você é um inútil, ninguém te
  suporta." ('You are useless, nobody can stand you.')

**6. Criticising what someone did, without insulting the person, is not Severe.**
- With an offensive tone, it is **Mild**: "Que ideia ridícula." ('What a ridiculous idea.')
- Without an offensive tone, it is **None**: "Acho que você errou essa conta." ('I think
  you got that sum wrong.')
