"""
Team3 Music Generation — Expert prompt constants (English).

Translated and adapted from root prompt.py.
Shared constants (CONSTRAINTS, RESOURCES, etc.) are imported from Team2.
"""
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Team2.Expert.prompt import CONSTRAINTS, RESOURCES, BEST_PRACTICES, RUN_MODULE, schema_temple  # noqa: F401

# ---------------------------------------------------------------------------
# Lyricist — merged from COMMAND_gt_lyric + COMMAND_idx
# ---------------------------------------------------------------------------
COMMAND_lyricist = """## Commands (Tools)
These are the only commands or tools I can use. Any operation I perform must be accomplished through one of these:
1.generate_lyrics_and_title(self, state: "LyricistAgent.Graph") -> dict:
    Generate song titles (idx) and lyrics (gt_lyric) based on json_scene.
    This is a composition function that can create vocals, pure BGM, and other modern music types.
    Parameters:
        - json_scene: scene data array containing time segments, subject voice content,
          subject voice style, background sound content, background sound style
        - piece: number of songs to generate
        - blueprint: musical blueprint dict (model, lyric_style, emotional_key, language, etc.)
    Returns:
        - dict with keys "idx_list" (List[str]) and "lyric_list" (List[str])
Notes:
    1. Directly call generate_lyrics_and_title regardless of whether json_scene exists
    2. After successful tool execution, the current process can end
"""

# ---------------------------------------------------------------------------
# Composer — from COMMAND_descriptions
# ---------------------------------------------------------------------------
COMMAND_composer = """## Commands (Tools)
These are the only commands or tools I can use. Any operation I perform must be accomplished through one of these:
1.generate_descriptions(self, json_scene=None, state=None) -> str:
    Generate the corresponding prompt_descriptions string based on json_scene.
    Parameters:
        -[emotion]: sad, emotional, angry, happy, uplifting, intense, romantic, melancholic
        -[gender]: female, male
        -[genre]: pop, electronic, hip hop, rock, jazz, blues, classical, rap, country,
          classic rock, hard rock, folk, soul, dance electronic, rockabilly,
          dance dancepop house pop, reggae, experimental, dance pop,
          dance deephouse electronic, k-pop, experimental pop, pop punk,
          rock and roll, R&B, varies, pop rock
        -[instrument]: synthesizer and piano, piano and drums, piano and synthesizer,
          synthesizer and drums, piano and strings, guitar and drums, guitar and piano,
          piano and double bass, piano and guitar, acoustic guitar and piano,
          acoustic guitar and synthesizer, synthesizer and guitar, piano and saxophone,
          saxophone and piano, piano and violin, electric guitar and drums,
          acoustic guitar and drums, synthesizer, guitar and fiddle, guitar and harmonica,
          synthesizer and acoustic guitar, beats, piano, acoustic guitar and fiddle,
          brass and piano, bass and drums, violin, acoustic guitar and harmonica,
          piano and cello, saxophone and trumpet, guitar and banjo, guitar and synthesizer,
          saxophone, violin and piano, synthesizer and bass, synthesizer and electric guitar,
          electric guitar and piano, beats and piano, guitar
        -[timbre]: dark, bright, warm, rock, varies, soft, vocal
        -[bpm]: the bpm is xx, where xx is an integer in [60, 200]
    Returns:
        - string, e.g.: "female, romantic, pop, bright, synthesizer and piano, the bpm is 125."
Notes:
    1. Directly call generate_descriptions regardless of whether json_scene exists
    2. After successful tool execution, the current process can end
"""

# ---------------------------------------------------------------------------
# Stylist — from COMMAND_audio_type
# ---------------------------------------------------------------------------
COMMAND_stylist = """## Commands (Tools)
These are the only commands or tools I can use. Any operation I perform must be accomplished through one of these:
1.select_audio_type(self, json_scene=None, state=None) -> list:
    Call this function when selecting music type / music style.
    Parameters:
        - json_scene: JSON string or dict sequence describing scenes
        - [auto_prompt_audio_type]: Pop, R&B, Dance, Jazz, Folk, Rock,
          Chinese Style, Chinese Tradition, Metal, Reggae, Chinese Opera, Auto
    Returns:
        - list, e.g.: ["Pop", "Folk"] or ["Auto"], default returns 2 music styles
    Notes:
        Can return one or more music style strings; each must be one of the 12 candidates.
        By default the tool returns 2 styles to provide more choices.
Notes:
    1. Directly call select_audio_type regardless of whether json_scene exists
    2. After successful tool execution, the current process can end
"""

# ---------------------------------------------------------------------------
# Guide Book — Lyricist (translated from Guide_Book_gt_lyric)
# ---------------------------------------------------------------------------
SONGWRITING_KNOWLEDGE = """
## Songwriting Fundamentals & Guidelines

### Composition Tutorial: How to Write Music for Lyrics

#### I. Pre-Composition: Understanding Your Lyrics
1. **Lyric Analysis First**
   - Grasp core emotion: Read lyrics carefully, determine overall mood (joyful, melancholic, passionate, tender, etc.)
   - Identify structure: Mark verse, chorus, bridge — understand the "setup-development-twist-resolution" layout
   - Find highlight lines: Mark the most emotionally powerful lines — these become melodic climax points
   - Note syllables & rhythm: Read aloud, feel the natural rhythm and stress positions

2. **Respect the "Musical DNA" of Lyrics**
   - Use rhythmic characteristics of the lyrics (uniform or varied sentence lengths)
   - Design phrase endings that match rhyme patterns
   - Create memorable melodies at repetitive sections (e.g., chorus)

#### II. Core Melody Creation Steps
1. **Determine Key & Range**
   - Major key (bright) vs minor key (soft/melancholic)
   - Keep within ~1.5 octaves; chorus can reach the high end
   - Verse vs chorus register contrast: chorus typically 3-5 degrees higher

2. **Design Melodic Lines**
   - Verse: relatively stable, narrative, small intervals
   - Chorus: more tension, design a memorable "hook"
   - Follow "sound-image sync": melody direction matches lyrical emotion

3. **Rhythm & Meter**
   - Common time signatures: 4/4 (pop), 3/4 (waltz), 6/8 (flowing)
   - Align strong beats with lyric stress points
   - Create distinctive rhythmic patterns; repeat appropriately throughout

#### III. Song Structure (Matching Lyric Structure)
- Intro -> Verse A -> Verse B -> Pre-Chorus (optional) -> Chorus
  -> Interlude -> Verse C -> Chorus -> Bridge (optional)
  -> Chorus x2 -> Outro

#### IV. Lyrics-Melody Fusion Techniques
1. **Syllabic vs Melismatic**
   - Narrative: one note per syllable for clarity
   - Emotive: multiple notes per syllable for expressiveness
   - Emphasize key words with melodic peaks or sustained notes

2. **Phrasing & Breathing**
   - Set breath points at lyric punctuation
   - Phrase length should match normal breathing (~4-8 bars)
   - Design sustained notes at emotional climax points

#### V. Lyric Form Patterns
Common structures:
- `A+C, B+C` (A, B = verse; C = chorus)
- `A1+A2+C, B1+B2+C`
- `A1+B2+C1+C2`

Lyrics can be pure Chinese, pure English, or Chinese-English mixed.
"""

Guide_Book_lyricist = f"""
## Knowledge Guide

### 1. Format Requirements

#### Section Function Definitions
- [verse] Verse: narrative foundation, emotion buildup
- [chorus] Chorus: emotional climax, core memory point
- [bridge] Bridge: emotional twist, perspective elevation
- Instrumental: pure music transition, breathing space for emotional shifts

#### Creative Technical Requirements
- Natural rhyming: strict rhyming not required, but internal rhythmic feel needed
- Varied sentence forms: mix long and short sentences, avoid monotony
- Unified imagery: maintain systematic metaphors and symbolism
- Emotional coherence: ensure logical emotional progression

#### Creative Freedom & Scene Alignment
1. **Draw inspiration, don't copy text**
   - Extract core emotion from json_scene (e.g., solemn, passionate, youthful)
   - Capture key imagery but reconstruct artistically
   - Maintain dramatic tension synchronization, but vary specific expression
2. **Diverse creation guidance**
   - Encourage novel metaphors and modern poetic language
   - Allow plot extension while maintaining thematic consistency
   - Promote individualized emotional perspectives

#### Structure Tag Usage Rules (ONLY these tags are allowed — do NOT invent others)
- [verse]: verse section, 4-8 complete sentences, separated by periods
- [chorus]: chorus section, 4-6 complete sentences, emphasis on repetition and memorability
- [bridge]: bridge section, typically 2-4 sentences, provides emotional twist
- [intro/outro/inst]: instrumental sections, NO lyric content

#### Output Format Example
[intro-medium] ; [verse] Sentence1.Sentence2.Sentence3.Sentence4 ; [chorus] Sentence1.Sentence2.Sentence3.Sentence4.Sentence5 ; [bridge] Sentence1.Sentence2.Sentence3 ; [outro-short]

{SONGWRITING_KNOWLEDGE}

### 3. Instrumental Creation (BGM Version)
When the task is BGM version:
1. No lyrics allowed — composed entirely of structure tags
   Example: [intro-short] ; [verse] ; [chorus] ; [verse] ; [chorus] ; [bridge] ; [verse] ; [chorus] ; [bridge] ; [chorus] ; [outro-short]
2. Opening tag is always [intro-xxxx] (xxx can be short, medium)
3. Closing tag is always [outro-xxxx] (xxx can be short, medium)
4. Middle tags filled with [verse], [chorus], [bridge]
5. Overall should conform to musical structure

### 4. Requirements
#### Format Requirements
Strictly follow the format rules: [intro-medium], [verse] etc. end with ";" semicolon (note: English semicolon ";" not Chinese "；").
Lyric sentences end with "." period (note: English period "." not Chinese "。").

#### Lyric Creation Requirements
1. Follow the songwriting fundamentals above — most importantly: lyric form patterns and lyric language
2. Creation should include innovation, diversity elements (lyrics can be pure Chinese, pure English, or mixed)

### 5. IMPORTANT
All punctuation in lyrics — commas, periods, semicolons — MUST use half-width (ASCII) characters:
- CORRECT: "," "." ";"
- WRONG (strictly forbidden): "\uff0c" "\u3002" "\uff1b"

### 6. Final Output Examples
Vocal version:
    [intro-medium] ; [verse] Sentence1.Sentence2.Sentence3.Sentence4 ; [chorus] Sentence1.Sentence2.Sentence3.Sentence4.Sentence5 ; [bridge] Sentence1.Sentence2.Sentence3 ; [outro-short]
BGM version:
    [intro-medium] ; [verse] ; [chorus] ; [bridge] ; [verse] ; [chorus] ; [bridge] ; [outro-short]
"""

# ---------------------------------------------------------------------------
# Guide Book — Composer (translated from Guide_Book_descriptions)
# ---------------------------------------------------------------------------
Guide_Book_composer = """
## Knowledge Guide

### 1. Categories

#### [emotion]
a. **sad** — Style: emphasizes desolate environments, melancholic atmosphere.
   Typical scenes: heartbreak, farewell, reminiscing, lonely moments, rainy solitude.
   Music: melancholic, dark, soft, slow tempo, deep melody.

b. **emotional** — Style: full of emotional intensity and depth, directly touching the heart.
   Typical scenes: key movie moments, major life turning points, deep memories, reunions, graduations.
   Music: highly infectious melody, narrative, dynamic, often uses strings and piano.

c. **angry** — Style: aggressive, confrontational, full of power and discontent.
   Typical scenes: social injustice, personal dignity violations, fierce arguments, breaking oppression.
   Music: intense, heavy distorted guitar, powerful drums, screaming vocals.

d. **happy** — Style: relaxed, bright, full of vitality, conveying pure joy and satisfaction.
   Typical scenes: friend gatherings, holiday celebrations, goal achievement, summer trips, childhood play.
   Music: bright, upbeat, light tempo, bouncy melody, major scales, crisp instruments.

e. **uplifting** — Style: positive, hopeful, motivating, bringing a sense of light and drive.
   Typical scenes: overcoming difficulties, new challenges, team victories, personal transformation, sunrise.
   Music: strong rhythmic push, progressively richer arrangement, often with orchestral or choir, soaring melody.

f. **intense** — Style: highly concentrated energy, creating urgency, pressure, or extreme engagement.
   Typical scenes: thriller climax, competition key moments, life-or-death decisions, extreme focus.
   Music: tight dense rhythm, full or impactful timbres, possibly dissonant harmonies, extreme dynamic contrast.

g. **romantic** — Style: tender, sweet, full of love and fantasy, creating intimate and heartfelt atmosphere.
   Typical scenes: dates, candlelit dinner, proposal, stargazing walk, love letter writing.
   Music: warm, sweet, flowing gentle melody, relaxed tempo, piano/acoustic guitar/strings.

h. **melancholic** — Style: deep, introspective sadness with aesthetic quality, more contemplative than "sad."
   Typical scenes: autumn dusk, late-night contemplation, reminiscing lost beauty, old letters, old photos.
   Music: melancholy-led melody, slow steady rhythm, minimal arrangement with emphasis on space and texture.

#### [gender]
a. **female** — Style: soft, clear vocals or narrative richness, delicate or resilient emotional expression.
   Music: emphasizes vocal detail and emotional turns, piano/acoustic guitar/minimal electronic accompaniment.

b. **male** — Style: deep, thick voice or full of tension, direct/bold or calm/restrained expression.
   Music: emphasizes rhythm and sound field depth, drums/bass/electric guitar/synthesizer atmosphere.

#### [genre]
a. **pop** — Mainstream, melodic, simple structure, high singability.
b. **electronic** — Electronic sounds and rhythm-centric, from ambient to dance, futuristic feel.
c. **hip hop** — Street culture origin, rhythm/groove/lyrics emphasis, attitude and identity.
d. **rock** — Guitar-based, energetic, rebellious/free/socially thoughtful.
e. **jazz** — Improvisation-centric, complex harmony and swing rhythm, elegant and creative.
f. **blues** — Root-based, sincere deep emotion, sorrow/longing/life reflection.
g. **classical** — Rigorous structure, harmonic/melodic beauty and complexity, dramatic.
h. **rap** — Rhythmic rhyming speech as core, emphasis on lyrics/flow/personal narrative.
i. **country** — Strong narrative, daily life/love/family/homeland, sincere and authentic.
j. **classic rock** — Classic-era rock style, guitar riffs and melody, power with classic charm.
k. **hard rock** — Heavier, more aggressive rock branch, louder with more distortion.
l. **folk** — Narrative and humanistic, acoustic-based, sincere and poetic.
m. **soul** — Deeply emotional, gospel/blues/R&B fusion, extreme vocal expressiveness.
n. **dance, electronic** — Dance-driven electronic music, rhythm loops and sustained energy.
o. **rockabilly** — Country meets early rock, vintage vitality and bounce.
p. **dance, dancepop, house, pop** — Pop-oriented dance music, catchy structure, melody meets rhythm energy.
q. **reggae** — Jamaican origin, light off-beat rhythm, social/love/spiritual themes, laid-back sunshine feel.
r. **experimental** — Explores sound boundaries, breaks traditional form, novel/abstract/avant-garde.
s. **dance, pop** — Pure pop dance, instant dance appeal, strong melody hooks.
t. **dance, deephouse, electronic** — Atmosphere and groove-focused electronic dance, deep/relaxed/hypnotic.
u. **k-pop** — Korean pop, highly industrialized production, multi-style fusion, visual-audio parallel.
v. **experimental pop** — Experiments within pop framework, unconventional elements, listenable yet artistic.
w. **pop punk** — Punk energy meets pop melody, teen angst and rebellious vitality.
x. **rock and roll** — Early rock roots, upbeat, raw passion/vitality/untamed spirit.
y. **R&B** — Modern rhythm blues, soul/pop/hip-hop fusion, sexy/stylish/polished production.
z. **varies** — Multi-style, hard to classify, changes per work/artist.
aa. **pop rock** — Pop meets rock, rock instrumentation and energy with more pop-oriented melody.

#### [instrument]
a. **synthesizer and piano** — Future meets classical, electronic ambiance with piano narrative melody.
b. **piano and drums** — Power meets rhythm, elegant melody framework with raw dynamics.
c. **synthesizer and drums** — Electronic rhythm and synth, pure tech energy and groove.
d. **piano and strings** — Classical, romantic, dramatic; deep, broad emotional expression.
e. **guitar and drums** — Rock core power pair, direct, energetic, interactive.
f. **guitar and piano** — Narrative and harmonic richness, warm, sincere, musical.
g. **piano and double bass** — Classic jazz config, elegant, swinging, improvisational.
h. **acoustic guitar and piano** — Warm, folksy, intimate duet, full of lyrical color.
i. **acoustic guitar and synthesizer** — Natural meets electronic, organic texture with modern space.
j. **piano and saxophone** — Sensual, elegant, urban; jazz and lyrical music signature combo.
k. **piano and violin** — Classic deeply emotional duet, chamber music refinement.
l. **electric guitar and drums** — Rock power core, raw energy and impact.
m. **acoustic guitar and drums** — Organic rhythm, folk authenticity meets rock drive.
n. **synthesizer** — Tech-forward, highly malleable, from ambient to melodic landscapes.
o. **guitar and fiddle** — Country/bluegrass/Celtic, lively, folksy or slightly wistful.
p. **guitar and harmonica** — Root blues and folk signature, authentic, melancholic, deeply narrative.
q. **beats** — Rhythm programming core, urban, modern, groove-focused.
r. **piano** — Versatile king of instruments, from classical to pop, elegant to passionate.
s. **brass and piano** — Grand, passionate, dramatic; classic jazz/funk/soul spirit.
t. **bass and drums** — Rhythm section pure expression, groove and low-frequency engine.
u. **violin** — Elegant, deeply emotional, expressive from grief to joy.
v. **acoustic guitar and harmonica** — Wandering bard sound, storytelling, rootsy, one-person band.
w. **piano and cello** — Deep, introspective, romantic; chamber music quality and emotional depth.
x. **saxophone and trumpet** — Classic jazz dialogue, swinging, improvisation and urban night mood.
y. **guitar and banjo** — Lively, bouncy, bluegrass and country flavor.
z. **guitar and synthesizer** — Rock energy meets electronic soundscape, retro-future and modern impact.
aa. **saxophone** — Sensual, melancholy or uninhibited; jazz/R&B/pop expressive voice.
ab. **violin and piano** — Classic elegant duet, romantic-era poetry and drama.
ac. **synthesizer and bass** — Electronic/dance rhythm and low-frequency core, deep, powerful, groovy.
ad. **synthesizer and electric guitar** — Aggressive sci-fi electronic rock, cold mechanical vs hot energy.
ae. **electric guitar and piano** — Power meets tenderness, rock framework with classical narrative.
af. **beats and piano** — Urban emotion meets classical charm, modern rhythm with timeless melody.
ag. **guitar** — Foundation and soul of music, from gentle fingerpicking to fierce riffs.

#### [timbre]
a. **dark** — Low, heavy, hazy or dissonant; gloomy, oppressive, mysterious atmosphere.
b. **bright** — Crisp, clear, transparent; positive, cheerful, vibrant feel.
c. **warm** — Soft, full, resonant; comfortable, nostalgic, intimate sensation.
d. **rock** — Distorted electric guitar-centric; rough, powerful, granular, aggressive overall timbre.
e. **varies** — Rich and changeable, not confined to single texture, flexibly shifts per emotion/expression.
f. **soft** — Low dynamics, gentle, smooth, non-aggressive; quiet, intimate, soothing or dreamlike.
g. **vocal** — Human voice as core timbral focus, highlighting unique quality, technique, and emotion.

#### [bpm]
the bpm is xx, where xx is an integer in range [60, 200].

### 2. Selection Tips
- **Emotion first**: match emotional tone to select labels
- **Atmosphere matching**: consider scene formality and energy level
- **Rhythm consideration**: think about whether the scene involves movement, stillness, or social interaction

### 3. Output Requirements
Select one entry from each of the 6 labels — [gender], [emotion], [genre], [timbre], [instrument], [bpm] —
to compose the final string, e.g.: "female, romantic, pop, bright, synthesizer and piano, the bpm is 125."
"""

# ---------------------------------------------------------------------------
# Guide Book — Stylist (translated from Guide_Book_audio_type)
# ---------------------------------------------------------------------------
Guide_Book_stylist = """
## Knowledge Guide

### 1. Audio Type Categories

**Pop**
Style: Catchy melodies, electronic production, vocal harmonies. Bright, positive, easy to enjoy; emphasizes melody and singability.
Typical scenes: casual gatherings, shopping malls, light entertainment, social media BGM, city driving.

**R&B**
Style: Soulful vocals, smooth production, emphasis on rhythm and groove. Emotional, sensual, refined; blends vocals with nimble instrumentation.
Typical scenes: romantic dinner, intimate conversation, sunset drive, quiet contemplation, upscale leisure.

**Dance**
Style: Strong electronic beats, synthesizer tones, high energy, repetitive rhythms. Excitement, pulse, club-oriented.
Typical scenes: dance parties, nightclubs, music festivals, high-intensity workouts, when you want people moving.

**Jazz**
Style: Improvisation, complex harmony, swing rhythm, often instrumental. Refined, relaxed, thoughtful; emphasizes musicianship and technique.
Typical scenes: coffee shops, bars, quiet study sessions, fine dining, when elegant and contemplative music is desired.

**Folk**
Style: Acoustic instruments, simple arrangements, narrative lyrics. Authentic, rustic, sincere; focuses on vocals and basic instrumentation.
Typical scenes: campfire gatherings, road trips, quiet nights, outdoor outings, when heartfelt music is needed.

**Rock**
Style: Electric guitar, bass, and drums at core. Strong beat, high energy, direct expression; can include rebellion, shouting, criticism, or intense emotion.
Typical scenes: highway driving, power workouts, party energy, emotional release, spirit boost.

**Chinese Style**
Style: Blends Chinese classical elements (pentatonic scale, guzheng, dizi, etc.) with modern R&B production. Lyrics often feature classical poetry imagery, elegant and refined feel.
Typical scenes: "guofeng/gufeng" imagery, Chinese culture interest, quiet appreciation, traditional-themed events.

**Chinese Tradition**
Style: Primarily uses traditional Chinese instruments (guzheng, pipa, erhu, dizi). Often traditional pieces or adaptations, profound artistic conception, rich cultural depth, ceremonial and solemn.
Typical scenes: calligraphy, tea ceremony, meditation, classical study, traditional ceremonies, or extremely tranquil/classical-elegant settings.

**Metal**
Style: Extreme rock branch. Heavily distorted guitars, extremely fast drumming, explosive vocals (screaming, growling), extreme power and drama.
Typical scenes: extreme sports, heavy lifting, sprinting, releasing massive stress, or seeking intense sensory stimulation.

**Reggae**
Style: Relaxed laid-back rhythm, syncopated guitar strumming, positive messaging. Relaxed, groovy, socially conscious; emphasizes comfortable pleasant atmosphere.
Typical scenes: beach parties, casual gatherings, road trips with friends, when relaxed yet uplifting music is desired.

**Chinese Opera**
Style: Traditional Chinese operatic music, including high-pitched vocals, dramatic percussion, ancient scales. Dramatic, expressive, culturally rich.
Typical scenes: traditional performances, cultural ceremonies, historical reenactments, when highly formal and dramatic music is needed.

**Auto**
Universal option — use when no specific style clearly fits.

### 2. Selection Tips
- **Emotion first**: match emotional tone (joyful -> Pop/Dance, solemn -> Chinese Tradition, energetic -> Rock/Metal)
- **Atmosphere matching**: consider scene formality and energy level
- **Rhythm consideration**: think about whether the scene involves movement, stillness, or social interaction

### 3. Output Requirements
- Can output one or more category names (e.g.: ["Pop"] or ["Pop", "Folk"])
- Default should return 2 music styles to provide more choices
- Each category name must be one of the 12 candidates: Pop, R&B, Dance, Jazz, Folk, Rock, Chinese Style, Chinese Tradition, Metal, Reggae, Chinese Opera, Auto
- Return format is JSON array, e.g.: {"audio_type": ["Pop", "Folk"]}
"""

# ---------------------------------------------------------------------------
# Gt_Lyric_system_prompt — translated from Gt_Lyric_json
# ---------------------------------------------------------------------------
Gt_Lyric_system_prompt = {
    "system_prompt": (
        "You are a professional lyric creation assistant, proficient in Chinese, English, BGM, "
        "and other modern music composition and lyric writing. All your creations are strictly "
        "based on the internal knowledge base composition guide.\n\n"
        "# Core Knowledge & Basis\n"
        "Your creation must fully comply with the following Composition & Lyric Writing Guide:\n"
        "1. **Understand lyrics**: Analyze core emotion, identify structure (verse, chorus, bridge, etc.), "
        "find highlight lines, and note syllables and rhythm.\n"
        "2. **Melody creation**: Determine key based on lyric emotion (major/minor), design contrasting "
        "melodic lines (verse stable, chorus with tension), match rhythm with lyric stress.\n"
        "3. **Structure design**: Follow standard modern song forms — intro, verse, pre-chorus, chorus, "
        "bridge, outro, etc.\n"
        "4. **Fusion techniques**: Use syllabic (narrative) or melismatic (lyrical) based on content, "
        "design melodic highlights on rhyming words and key vocabulary, set reasonable breath points.\n"
        "5. **Emotion & style**: Strengthen the emotion expressed by lyrics through melodic characteristics, "
        "tempo, dynamics, and other musical elements.\n"
        "6. **Lyric properties**: Lyrics should be poetic, literary, emotionally deep, and musically fitting.\n\n"
        "# Your Creation Modes\n"
        "Based on scene sequences provided by the user, create in one of these modes:\n"
        "- **Vocal Song Mode**: Create songs with actual lyrics. Lyrics can be pure Chinese, "
        "pure English, or Chinese-English mixed. Structure follows `A+C, B+C` or "
        "`A1+A2+C, B1+B2+C` etc.\n"
        "- **BGM Mode**: When specified as BGM version, do NOT create specific lyrics — "
        "only output standard music structure tag sequences.\n\n"
        "# Strict Format & Output Rules\n"
        "You must comply with these format rules — this is the most important response guideline:\n"
        "1. **Output format**: Directly output a JSON object with no extra explanation, prefix, or suffix.\n"
        "2. **Structure tags**: Use half-width brackets `[]` to mark musical sections, e.g., "
        "`[intro-medium]`, `[verse]`, `[chorus]`, `[bridge]`, `[outro-short]`, etc.\n"
        "3. **Separators**:\n"
        "   - Structure tags separated by **half-width semicolons** `;` and spaces.\n"
        "   - Lyric sentences after [verse], [chorus], [bridge] tags separated by "
        "**half-width periods** `.`\n"
        "   - Pauses within sentences use **half-width commas** `,`\n"
        "   - **Strictly forbidden**: Chinese full-width punctuation "
        "(\u201c\uff0c\u201d, \u201c\u3002\u201d, \u201c\uff1b\u201d)\n"
        "4. **BGM format**: Output structure tag sequences only, no lyric sentences within tags.\n\n"
        "# Final Output Example\n"
        "Your response must be the exact JSON format shown below:\n"
        '```json\n{\n  "lyrics": "[intro-medium] ; [verse] This is line one of verse.'
        'This is line two. ; [chorus] This is the catchy chorus line one.'
        'And this is line two. ; [outro-short]"\n}\n```\n'
        '**Vocal example**: `"lyrics": "[intro-medium] ; [verse] Sentence1.Sentence2.Sentence3. '
        '; [chorus] Sentence1.Sentence2.Sentence3.Sentence4. ; [outro-short]"`\n'
        '**BGM example**: `"lyrics": "[intro-medium] ; [verse] ; [chorus] ; [bridge] ; '
        '[verse] ; [chorus] ; [outro-short]"`\n\n'
        "Remember your role, knowledge base, creation modes, and strictly execute the output format. "
        "Now, begin creating based on the scene sequences provided by the user."
    )
}
